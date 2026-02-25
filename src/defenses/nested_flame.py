import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy

try:
    import hdbscan
except ImportError:
    print("Please install hdbscan: pip install hdbscan")
    hdbscan = None

from ..fl.baseclient import BenignClient
from ..fl.baseserver import FedAvgAggregator

class FlameNestedServer(FedAvgAggregator):
    """
    Implements the FLAME defense mechanism with two-step anomaly detection
    to reduce false positives.
    """
    def __init__(self, model: torch.nn.Module, testloader: DataLoader = None,
                 device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)
        if hdbscan is None:
            raise ImportError("hdbscan is not installed. Please install it to use FlameServer.")

        self.config = config if config is not None else {}
        self.lamda = self.config.get('flame_lamda', 0.001)
        self.eta = self.config.get('flame_eta', 1.0)
        self.temporal_tau = self.config.get('temporal_tau', 3)  # rounds before confirming malicious
        self.sim_threshold = self.config.get('similarity_threshold', 0.95)  # cosine similarity for second step

        self.temporal_flags = {}  # stores how many times each client was flagged

        print(f"Initialized FlameServer with lamda={self.lamda}, eta={self.eta}, temporal_tau={self.temporal_tau}")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        if not self.received_params:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        malicious_indices, benign_indices, euclidean_distances = self.detect_anomalies()

        # --- Temporal mitigation ---
        final_malicious = []
        final_benign = []
        for i in range(len(self.received_params)):
            if i in malicious_indices:
                self.temporal_flags[i] = self.temporal_flags.get(i, 0) + 1
                if self.temporal_flags[i] >= self.temporal_tau:
                    final_malicious.append(i)
                else:
                    final_benign.append(i)  # temporarily keep benign
            else:
                self.temporal_flags[i] = 0
                final_benign.append(i)

        print(f"Flame filtered clients: {len(final_malicious)} malicious, {len(final_benign)} benign.")

        if not final_benign:
            print("Warning: Flame filtered all clients. Global model will not be updated.")
            self.received_params.clear()
            self.received_lens.clear()
            return self.get_params()

        # --- Robust Aggregation ---
        benign_distances = [euclidean_distances[i] for i in final_benign]
        clip_norm = torch.median(torch.tensor(benign_distances)).item() if benign_distances else 1.0
        weight_accumulator = {name: torch.zeros_like(param) for name, param in self.model.state_dict().items()}
        global_params_cpu = self.get_params()

        for idx in final_benign:
            local_params = self.received_params[idx]
            weight = 1.0 / len(final_benign)

            for name, param in local_params.items():
                if name.endswith('num_batches_tracked'):
                    continue
                diff = param.to(self.device) - global_params_cpu[name].to(self.device)
                if euclidean_distances[idx] > clip_norm:
                    diff *= clip_norm / euclidean_distances[idx]
                weight_accumulator[name].add_(diff * weight)

        final_state_dict = self.model.state_dict()
        for name, param in final_state_dict.items():
            if name in weight_accumulator:
                param.add_(weight_accumulator[name] * self.eta)
                if 'weight' in name or 'bias' in name:
                    std_dev = self.lamda * clip_norm
                    noise = torch.normal(0, std_dev, param.shape, device=param.device)
                    param.add_(noise)

        self.model.load_state_dict(final_state_dict)
        self.received_params.clear()
        self.received_lens.clear()
        return self.get_params()

    def detect_anomalies(self) -> Tuple[List[int], List[int], List[float]]:
        num_clients = len(self.received_params)
        if num_clients < 2:
            return [], list(range(num_clients)), []

        last_layer_names = self._get_last_layers(self.received_params[0])
        all_client_weights = []
        euclidean_distances = []
        global_params_cpu = self.get_params()

        for local_params in self.received_params:
            flat_update_diff = []
            for name, param in local_params.items():
                if 'weight' in name or 'bias' in name:
                    diff = param.to(self.device) - global_params_cpu[name].to(self.device)
                    flat_update_diff.append(diff.flatten())
            euclidean_distances.append(torch.linalg.norm(torch.cat(flat_update_diff)).item())

            last_layer_weights = []
            for name in last_layer_names:
                if name in local_params:
                    last_layer_weights.append(local_params[name].cpu().flatten())
            all_client_weights.append(torch.cat(last_layer_weights).numpy().astype(np.float64))

        client_weights_array = np.array(all_client_weights, dtype=np.float64)

        # --- First-step HDBSCAN clustering ---
        clusterer = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=max(2, num_clients // 2 + 1),
            allow_single_cluster=True
        )
        labels = clusterer.fit_predict(client_weights_array)

        benign_indices = []
        if np.all(labels == -1) or len(np.unique(labels)) == 1:
            benign_indices = list(range(num_clients))
        else:
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                largest_cluster_label = unique_labels[np.argmax(counts)]
                benign_indices = [i for i, label in enumerate(labels) if label == largest_cluster_label]
            else:
                benign_indices = list(range(num_clients))

        # Clients outside the largest cluster
        all_indices = set(range(num_clients))
        filtered_indices = list(all_indices - set(benign_indices))

        # --- Second-step: re-evaluate filtered clients ---
        for i in filtered_indices[:]:
            local_weights = client_weights_array[i]
            global_last_layer = np.concatenate([global_params_cpu[name].cpu().flatten().numpy() for name in last_layer_names])
            cos_sim = np.dot(local_weights, global_last_layer) / (np.linalg.norm(local_weights) * np.linalg.norm(global_last_layer) + 1e-8)
            if cos_sim >= self.sim_threshold:
                benign_indices.append(i)
                filtered_indices.remove(i)

        return filtered_indices, benign_indices, euclidean_distances

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        layer_names = list(state_dict.keys())
        param_layers = [name for name in layer_names if 'weight' in name or 'bias' in name]
        return param_layers[-2:]
