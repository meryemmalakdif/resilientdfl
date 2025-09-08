import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy

# Flame requires hdbscan for clustering
try:
    import hdbscan
except ImportError:
    print("Please install hdbscan: pip install hdbscan")
    hdbscan = None

from ..fl.baseserver import FedAvgAggregator

class FlameServer(FedAvgAggregator):
    """
    Implements the FLAME defense mechanism.

    This server combines dynamic clustering to filter malicious clients with
    adaptive clipping and noising for robust aggregation.
    """
    def __init__(self, model: torch.nn.Module, testloader: DataLoader = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)

        if hdbscan is None:
            raise ImportError("hdbscan is not installed. Please install it to use FlameServer.")

        self.config = config if config is not None else {}
        
        # FLAME-specific parameters from the config
        self.lamda = self.config.get('flame_lamda', 0.001) # Noise coefficient
        self.eta = self.config.get('flame_eta', 1.0) # Server learning rate/update scale
        
        self.global_model_params = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        print(f"Initialized FlameServer with lamda={self.lamda}, eta={self.eta}")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Filters malicious updates using FLAME's clustering and then performs
        robust aggregation with clipping and adaptive noise.
        """
        if not self.updates:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        malicious_indices, benign_indices, euclidean_distances = self.detect_anomalies()
        
        print(f"Flame detected {len(malicious_indices)} anomalous clients.")
        if malicious_indices:
            print(f"Filtering out clients at indices: {malicious_indices}")

        benign_updates = [self.updates[i] for i in benign_indices]
        
        if not benign_updates:
            print("Warning: Flame filtered out all clients. Global model will not be updated.")
            self.updates.clear()
            return self.get_params()

        # --- Robust Aggregation with Clipping and Noise ---
        benign_distances = [euclidean_distances[i] for i in benign_indices]
        clip_norm = torch.median(torch.tensor(benign_distances)).item() if benign_distances else 1.0

        weight_accumulator = {name: torch.zeros_like(param) for name, param in self.model.state_dict().items()}

        for i, original_index in enumerate(benign_indices):
            update = self.updates[original_index]
            weight = 1.0 / len(benign_updates) # Simple average over benign clients

            for name, param in update['weights'].items():
                if name.endswith('num_batches_tracked'): continue
                
                diff = param.to(self.device) - self.global_model_params[name].to(self.device)
                
                # Apply clipping
                if euclidean_distances[original_index] > clip_norm:
                    diff *= clip_norm / euclidean_distances[original_index]
                
                weight_accumulator[name].add_(diff * weight)

        # Update global model parameters
        final_state_dict = self.model.state_dict()
        for name, param in final_state_dict.items():
            if name in weight_accumulator:
                # Apply aggregated update with server learning rate
                param.add_(weight_accumulator[name] * self.eta)

                # Add adaptive noise
                if 'weight' in name or 'bias' in name:
                    std_dev = self.lamda * clip_norm
                    noise = torch.normal(0, std_dev, param.shape, device=param.device)
                    param.add_(noise)
        
        self.model.load_state_dict(final_state_dict)
        self.updates.clear()
        self.global_model_params = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        return self.get_params()

    def detect_anomalies(self) -> Tuple[List[int], List[int], List[float]]:
        """
        Detects anomalies using cosine similarity clustering on last layer weights.
        Returns (malicious_indices, benign_indices, all_euclidean_distances).
        """
        num_clients = len(self.updates)
        if num_clients < 2:
            return [], list(range(num_clients)), []

        last_layer_names = self._get_last_layers(self.updates[0]['weights'])
        
        all_client_weights = []
        euclidean_distances = []

        for update in self.updates:
            # Calculate Euclidean distance (update norm) for clipping
            flat_update_diff = []
            for name, param in update['weights'].items():
                if 'weight' in name or 'bias' in name:
                    diff = param.to(self.device) - self.global_model_params[name].to(self.device)
                    flat_update_diff.append(diff.flatten())
            euclidean_distances.append(torch.linalg.norm(torch.cat(flat_update_diff)).item())

            # Extract last layer weights for clustering
            last_layer_weights = []
            for name in last_layer_names:
                if name in update['weights']:
                     last_layer_weights.append(update['weights'][name].cpu().flatten())
            all_client_weights.append(torch.cat(last_layer_weights).numpy().astype(np.float64))
        
        # Perform clustering on last layer weights
        clusterer = hdbscan.HDBSCAN(
            metric="cosine",
            min_cluster_size=max(2, num_clients // 2 + 1), # Ensure majority cluster
            allow_single_cluster=True
        )
        labels = clusterer.fit_predict(np.array(all_client_weights))

        benign_indices = []
        if np.all(labels == -1) or len(np.unique(labels)) == 1:
            # If all are outliers or in a single cluster, assume all are benign
            benign_indices = list(range(num_clients))
        else:
            # Find the largest cluster
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                largest_cluster_label = unique_labels[np.argmax(counts)]
                benign_indices = [i for i, label in enumerate(labels) if label == largest_cluster_label]
            else: # All are outliers
                 benign_indices = list(range(num_clients))


        all_indices = set(range(num_clients))
        malicious_indices = list(all_indices - set(benign_indices))

        return malicious_indices, benign_indices, euclidean_distances

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """Get names of last two layers with parameters."""
        layer_names = list(state_dict.keys())
        param_layers = [name for name in layer_names if 'weight' in name or 'bias' in name]
        return param_layers[-2:]
