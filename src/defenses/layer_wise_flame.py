# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from typing import Dict, List, Optional, Tuple
# import numpy as np
# import copy

# # Flame requires hdbscan for clustering
# try:
#     import hdbscan
# except ImportError:
#     print("Please install hdbscan: pip install hdbscan")
#     hdbscan = None

# from ..fl.baseclient import BenignClient
# from ..fl.baseserver import FedAvgAggregator

# class LayerFlameServer(FedAvgAggregator):
#     """
#     FLAME server with temporal mitigation and layer-wise anomaly detection.
#     """
#     def __init__(self, model: torch.nn.Module, testloader: DataLoader = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
#         super().__init__(model, testloader, device)

#         if hdbscan is None:
#             raise ImportError("hdbscan is not installed. Please install it to use FlameServer.")

#         self.config = config if config else {}
#         self.lamda = self.config.get('flame_lamda', 0.001)
#         self.eta = self.config.get('flame_eta', 1.0)
#         self.tau = self.config.get('flame_tau', 3)  # temporal mitigation threshold

#         # Tracks how many rounds each client has been flagged as malicious
#         self.malicious_count = dict()

#         print(f"Initialized FlameServer with lamda={self.lamda}, eta={self.eta}, tau={self.tau}")

#     def aggregate(self) -> Dict[str, torch.Tensor]:
#         if not self.received_params:
#             print("Warning: No updates to aggregate.")
#             return self.get_params()

#         malicious_indices, benign_indices, euclidean_distances = self.detect_anomalies()

#         # --- Temporal mitigation ---
#         final_malicious_indices = []
#         for idx in malicious_indices:
#             self.malicious_count[idx] = self.malicious_count.get(idx, 0) + 1
#             if self.malicious_count[idx] >= self.tau:
#                 final_malicious_indices.append(idx)

#         # Update benign indices accordingly
#         final_benign_indices = [i for i in range(len(self.received_params)) if i not in final_malicious_indices]

#         print(f"FLAME detected {len(malicious_indices)} suspicious clients this round.")
#         print(f"{len(final_malicious_indices)} clients filtered after temporal mitigation.")
#         if final_malicious_indices:
#             print(f"Filtered client indices: {final_malicious_indices}")

#         if not final_benign_indices:
#             print("Warning: All clients filtered. Global model will not be updated.")
#             self.received_params.clear()
#             self.received_lens.clear()
#             return self.get_params()

#         # --- Robust aggregation ---
#         benign_distances = [euclidean_distances[i] for i in final_benign_indices]
#         clip_norm = torch.median(torch.tensor(benign_distances)).item() if benign_distances else 1.0

#         weight_accumulator = {name: torch.zeros_like(param) for name, param in self.model.state_dict().items()}
#         global_params_cpu = self.get_params()

#         for idx in final_benign_indices:
#             local_params = self.received_params[idx]
#             weight = 1.0 / len(final_benign_indices)

#             for name, param in local_params.items():
#                 if name.endswith('num_batches_tracked'): continue
#                 diff = param.to(self.device) - global_params_cpu[name].to(self.device)

#                 # Clip updates
#                 if euclidean_distances[idx] > clip_norm:
#                     diff *= clip_norm / euclidean_distances[idx]

#                 weight_accumulator[name].add_(diff * weight)

#         final_state_dict = self.model.state_dict()
#         for name, param in final_state_dict.items():
#             if name in weight_accumulator:
#                 param.add_(weight_accumulator[name] * self.eta)
#                 if 'weight' in name or 'bias' in name:
#                     std_dev = self.lamda * clip_norm
#                     noise = torch.normal(0, std_dev, param.shape, device=param.device)
#                     param.add_(noise)

#         self.model.load_state_dict(final_state_dict)
#         self.received_params.clear()
#         self.received_lens.clear()
#         return self.get_params()

#     def detect_anomalies(self) -> Tuple[List[int], List[int], List[float]]:
#         """
#         Layer-wise anomaly detection using HDBSCAN clustering.
#         Returns (malicious_indices, benign_indices, euclidean_distances)
#         """
#         num_clients = len(self.received_params)
#         if num_clients < 2:
#             return [], list(range(num_clients)), []

#         layer_names = self._get_all_layers(self.received_params[0])
#         euclidean_distances = []

#         # Store all client vectors per layer
#         all_layer_vectors = {layer: [] for layer in layer_names}

#         global_params_cpu = self.get_params()

#         for local_params in self.received_params:
#             flat_diff = []
#             for layer in layer_names:
#                 diff = local_params[layer].to(self.device) - global_params_cpu[layer].to(self.device)
#                 flat_diff.append(diff.flatten())
#                 all_layer_vectors[layer].append(local_params[layer].cpu().flatten())
#             euclidean_distances.append(torch.linalg.norm(torch.cat(flat_diff)).item())

#         # Perform HDBSCAN clustering per layer
#         layer_labels = {}
#         for layer, vectors in all_layer_vectors.items():
#             client_array = np.array(vectors, dtype=np.float64)
#             clusterer = hdbscan.HDBSCAN(
#                 metric="cosine",
#                 algorithm="generic",   # <-- ADD THIS
#                 min_cluster_size=max(2, num_clients // 2 + 1),
#                 allow_single_cluster=True
#             )
#             labels = clusterer.fit_predict(client_array)
#             layer_labels[layer] = labels

#         # Combine per-layer results
#         suspicious_scores = np.zeros(num_clients, dtype=int)
#         for layer, labels in layer_labels.items():
#             # Clients not in largest cluster or noise get +1
#             if np.all(labels == -1) or len(np.unique(labels)) == 1:
#                 continue  # all benign
#             unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
#             if len(unique_labels) > 0:
#                 largest_label = unique_labels[np.argmax(counts)]
#                 for idx, label in enumerate(labels):
#                     if label != largest_label:
#                         suspicious_scores[idx] += 1

#         # Treat clients flagged in at least one layer as suspicious
#         malicious_indices = [idx for idx, score in enumerate(suspicious_scores) if score > 0]
#         benign_indices = [i for i in range(num_clients) if i not in malicious_indices]

#         return malicious_indices, benign_indices, euclidean_distances

#     def _get_all_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
#         """Return names of all layers with parameters (weights and biases)."""
#         return [name for name in state_dict.keys() if 'weight' in name or 'bias' in name]



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


class LayerFlameServer(FedAvgAggregator):
    """
    FLAME defense with a second-pass False Positive Rescue (FPR) stage.

    Original FLAME problem:
      HDBSCAN clusters in cosine space. In non-IID FL, benign clients with
      minority label distributions land outside the largest cluster and get
      incorrectly flagged — ~45% false positive rate observed.

    Fix (without replacing FLAME's core logic):
      After FLAME identifies suspected malicious clients, run a second pass
      that re-examines each flagged client using influence alignment:

        s_i = cos(update_i, benign_cluster_average)

      Interpretation:
        s_i > threshold  → update aligns with what benign clients are doing
                           → likely a false positive → reinstate
        s_i <= threshold → update opposes or diverges from benign direction
                           → likely truly malicious → discard

    This preserves FLAME's clustering exactly as-is, and only adds a
    rescue step for clients FLAME flagged but didn't actually harm the model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        testloader: DataLoader = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__(model, testloader, device)
        if hdbscan is None:
            raise ImportError("hdbscan is not installed.")

        self.config = config if config is not None else {}
        self.lamda = self.config.get("flame_lamda", 0.001)
        self.eta = self.config.get("flame_eta", 1.0)

        # Second-pass threshold: flagged clients with alignment > this are reinstated
        # 0.0 means "neutral or positive alignment → probably benign"
        # Tune lower (e.g. -0.2) to be more permissive, higher (0.3) to be stricter
        self.fpr_threshold = self.config.get("flame_fpr_threshold", 0.0)

        print(
            f"Initialized FlameServer (+ FP Rescue) | "
            f"lamda={self.lamda} eta={self.eta} fpr_threshold={self.fpr_threshold}"
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> Dict[str, torch.Tensor]:
        if not self.received_params:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        # ── Pass 1: original FLAME clustering ────────────────────────────
        malicious_indices, benign_indices, euclidean_distances = self.detect_anomalies()
        print(f"  [FLAME Pass 1] Flagged {len(malicious_indices)} clients: {malicious_indices}")

        # ── Pass 2: false positive rescue ────────────────────────────────
        if malicious_indices and benign_indices:
            rescued, confirmed_malicious = self._rescue_false_positives(
                malicious_indices, benign_indices, euclidean_distances
            )
            if rescued:
                print(f"  [FP Rescue]    Reinstated {len(rescued)} false positives: {rescued}")
            if confirmed_malicious:
                print(f"  [FP Rescue]    Confirmed malicious: {confirmed_malicious}")
            benign_indices = sorted(benign_indices + rescued)
            malicious_indices = confirmed_malicious
        else:
            print("  [FP Rescue]    No flagged clients to examine.")

        print(f"  [FLAME Final]  Using {len(benign_indices)}/{len(self.received_params)} clients.")

        if not benign_indices:
            print("Warning: All clients filtered. Model not updated.")
            self.received_params.clear()
            self.received_lens.clear()
            return self.get_params()

        # ── Aggregation with clipping + noise (original FLAME) ───────────
        benign_distances = [euclidean_distances[i] for i in benign_indices]
        clip_norm = torch.median(torch.tensor(benign_distances)).item() if benign_distances else 1.0

        weight_accumulator = {
            name: torch.zeros_like(param)
            for name, param in self.model.state_dict().items()
        }
        global_params_cpu = self.get_params()

        for idx in benign_indices:
            local_params = self.received_params[idx]
            weight = 1.0 / len(benign_indices)
            for name, param in local_params.items():
                if name.endswith("num_batches_tracked"):
                    continue
                diff = param.to(self.device) - global_params_cpu[name].to(self.device)
                if euclidean_distances[idx] > clip_norm:
                    diff *= clip_norm / euclidean_distances[idx]
                weight_accumulator[name].add_(diff * weight)

        final_state_dict = self.model.state_dict()
        for name, param in final_state_dict.items():
            if name in weight_accumulator:
                param.add_(weight_accumulator[name] * self.eta)
                if "weight" in name or "bias" in name:
                    std_dev = self.lamda * clip_norm
                    noise = torch.normal(0, std_dev, param.shape, device=param.device)
                    param.add_(noise)

        self.model.load_state_dict(final_state_dict)
        self.received_params.clear()
        self.received_lens.clear()
        return self.get_params()

    # ------------------------------------------------------------------
    # Second-pass: False Positive Rescue
    # ------------------------------------------------------------------

    def _rescue_false_positives(
        self,
        malicious_indices: List[int],
        benign_indices: List[int],
        euclidean_distances: List[float],
    ) -> Tuple[List[int], List[int]]:
        """
        Re-examine FLAME's flagged clients using influence alignment.

        For each flagged client, compute cosine similarity between its
        gradient update and the average update of FLAME's benign cluster.

        If alignment > fpr_threshold → this client is pointing in the same
        direction as the benign majority → false positive → reinstate.

        If alignment <= fpr_threshold → update diverges from or opposes
        benign direction → confirm as malicious.

        Returns:
            rescued:            indices to move back to benign set
            confirmed_malicious: indices to keep as malicious
        """
        global_params = self.get_params()

        # Compute flat gradient vectors for all clients
        def get_flat_update(local_params: Dict) -> torch.Tensor:
            parts = []
            for name, param in local_params.items():
                if name.endswith("num_batches_tracked"):
                    continue
                diff = param.to(self.device) - global_params[name].to(self.device)
                parts.append(diff.flatten())
            return torch.cat(parts)

        # Average update of FLAME's benign cluster → reference direction
        benign_updates = [get_flat_update(self.received_params[i]) for i in benign_indices]
        benign_avg = torch.stack(benign_updates).mean(dim=0)
        benign_avg_norm = torch.linalg.norm(benign_avg).item()

        if benign_avg_norm < 1e-10:
            # Can't compute alignment, reinstate everyone to be safe
            return list(malicious_indices), []

        rescued = []
        confirmed_malicious = []

        for idx in malicious_indices:
            flagged_update = get_flat_update(self.received_params[idx])
            flagged_norm = torch.linalg.norm(flagged_update).item()

            if flagged_norm < 1e-10:
                # Zero update → neutral, reinstate
                rescued.append(idx)
                continue

            alignment = (
                torch.dot(flagged_update, benign_avg) / (flagged_norm * benign_avg_norm)
            ).item()

            if alignment > self.fpr_threshold:
                rescued.append(idx)
            else:
                confirmed_malicious.append(idx)

        return rescued, confirmed_malicious

    # ------------------------------------------------------------------
    # Original FLAME clustering (unchanged)
    # ------------------------------------------------------------------

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
                if "weight" in name or "bias" in name:
                    diff = param.to(self.device) - global_params_cpu[name].to(self.device)
                    flat_update_diff.append(diff.flatten())
            euclidean_distances.append(
                torch.linalg.norm(torch.cat(flat_update_diff)).item()
            )
            last_layer_weights = []
            for name in last_layer_names:
                if name in local_params:
                    last_layer_weights.append(local_params[name].cpu().flatten())
            all_client_weights.append(
                torch.cat(last_layer_weights).numpy().astype(np.float64)
            )

        client_weights_array = np.array(all_client_weights, dtype=np.float64)
        clusterer = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=max(2, num_clients // 2 + 1),
            allow_single_cluster=True,
        )
        labels = clusterer.fit_predict(client_weights_array)

        if np.all(labels == -1) or len(np.unique(labels)) == 1:
            benign_indices = list(range(num_clients))
        else:
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                largest_cluster_label = unique_labels[np.argmax(counts)]
                benign_indices = [
                    i for i, label in enumerate(labels)
                    if label == largest_cluster_label
                ]
            else:
                benign_indices = list(range(num_clients))

        all_indices = set(range(num_clients))
        malicious_indices = list(all_indices - set(benign_indices))
        return malicious_indices, benign_indices, euclidean_distances

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        layer_names = list(state_dict.keys())
        param_layers = [name for name in layer_names if "weight" in name or "bias" in name]
        return param_layers[-2:]