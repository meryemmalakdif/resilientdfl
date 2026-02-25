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


class FlameServer1(FedAvgAggregator):
    """
    FLAME defense with delta-based clustering to reduce false positives on
    non-IID data (e.g. FEMNIST).

    Key change — cluster on normalised UPDATE DELTAS, not raw weights:
        Raw weights vary legitimately across clients on non-IID data because
        each client's local distribution pushes weights to a different region
        of weight-space. HDBSCAN then mistakes data-heterogeneity for anomaly,
        producing constant false positives on benign rounds.
        The update delta  Δ = w_local - w_global  measures *direction of intent*:
        benign clients push toward better task performance (similar directions
        once normalised); poisoned clients push toward the backdoor manifold.
        L2-normalising each Δ removes the magnitude confound from different
        local dataset sizes, leaving HDBSCAN with a clean directional signal.

    Logging:
        Set  server.received_ids        = [client_id, ...]  (same order as received_params)
        Set  server.known_malicious_ids = {id, ...}          (from config or experiment runner)
        The server will then report truly-malicious vs false-positive counts each round.
        Without these, it falls back to printing raw filtered indices.
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
            raise ImportError(
                "hdbscan is not installed. Please install it to use FlameServer."
            )

        self.config = config if config is not None else {}
        self.lamda = self.config.get("flame_lamda", 0.001)
        self.eta   = self.config.get("flame_eta",   1.0)

        # False-positive guards
        self.max_filter_fraction = self.config.get("flame_max_filter_fraction", 0.3)
        self.min_cluster_ratio   = self.config.get("flame_min_cluster_ratio",   2.0)

        # Optional ground-truth for FP/TP logging
        self.known_malicious_ids: set = set(self.config.get("known_malicious_ids", []))
        self.received_ids: List[int] = []   # caller populates before each aggregate()

        print(
            f"Initialized FlameServer | lamda={self.lamda}, eta={self.eta}, "
            f"max_filter_fraction={self.max_filter_fraction}, "
            f"min_cluster_ratio={self.min_cluster_ratio}"
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> Dict[str, torch.Tensor]:
        if not self.received_params:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        malicious_indices, benign_indices, euclidean_distances = self.detect_anomalies()

        # --- Filtering report ---
        print(f"FLAME filtered {len(malicious_indices)}/{len(self.received_params)} client(s).")
        if malicious_indices:
            if self.received_ids and self.known_malicious_ids:
                truly_malicious = sorted([
                    self.received_ids[i] for i in malicious_indices
                    if self.received_ids[i] in self.known_malicious_ids
                ])
                false_positives = sorted([
                    self.received_ids[i] for i in malicious_indices
                    if self.received_ids[i] not in self.known_malicious_ids
                ])
                print(
                    f"  Truly malicious removed : {truly_malicious} ({len(truly_malicious)})\n"
                    f"  False positives removed : {false_positives} ({len(false_positives)})"
                )
            else:
                # Ground-truth not available — just show indices
                print(f"  Filtered indices: {sorted(malicious_indices)}")

        if not benign_indices:
            print("Warning: FLAME filtered all clients. Global model unchanged.")
            self.received_params.clear()
            self.received_lens.clear()
            return self.get_params()

        # --- Robust aggregation: clip + adaptive noise ---
        benign_distances = [euclidean_distances[i] for i in benign_indices]
        clip_norm = torch.median(torch.tensor(benign_distances)).item() if benign_distances else 1.0

        weight_accumulator = {
            name: torch.zeros_like(param)
            for name, param in self.model.state_dict().items()
        }
        global_params_cpu = self.get_params()

        for original_index in benign_indices:
            local_params = self.received_params[original_index]
            weight = 1.0 / len(benign_indices)

            for name, param in local_params.items():
                if name.endswith("num_batches_tracked"):
                    continue
                diff = param.to(self.device) - global_params_cpu[name].to(self.device)
                if euclidean_distances[original_index] > clip_norm:
                    diff *= clip_norm / euclidean_distances[original_index]
                weight_accumulator[name].add_(diff * weight)

        final_state_dict = self.model.state_dict()
        for name, param in final_state_dict.items():
            if name in weight_accumulator:
                param.add_(weight_accumulator[name] * self.eta)
                if "weight" in name or "bias" in name:
                    noise = torch.normal(0, self.lamda * clip_norm, param.shape, device=param.device)
                    param.add_(noise)

        self.model.load_state_dict(final_state_dict)
        self.received_params.clear()
        self.received_lens.clear()
        return self.get_params()

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalies(self) -> Tuple[List[int], List[int], List[float]]:
        """
        Cluster on L2-normalised last-layer update deltas (Δ = w_local - w_global).
        Returns (malicious_indices, benign_indices, all_euclidean_distances).
        """
        num_clients = len(self.received_params)
        if num_clients < 2:
            return [], list(range(num_clients)), []

        last_layer_names = self._get_last_layers(self.received_params[0])
        global_params_cpu = self.get_params()

        all_client_deltas = []
        euclidean_distances = []

        for local_params in self.received_params:
            # Full-model Euclidean distance — used for clipping in aggregation
            full_delta_parts = [
                (local_params[n].to(self.device) - global_params_cpu[n].to(self.device)).flatten()
                for n in local_params
                if "weight" in n or "bias" in n
            ]
            euclidean_distances.append(torch.linalg.norm(torch.cat(full_delta_parts)).item())

            # Last-layer delta — clustering feature
            last_layer_delta = torch.cat([
                (local_params[n].cpu() - global_params_cpu[n].cpu()).flatten()
                for n in last_layer_names if n in local_params
            ]).to(torch.float64)

            # L2-normalise: cosine distance then measures direction only
            norm = torch.linalg.norm(last_layer_delta)
            if norm > 1e-10:
                last_layer_delta = last_layer_delta / norm

            all_client_deltas.append(last_layer_delta.numpy())

        client_deltas_array = np.array(all_client_deltas, dtype=np.float64)

        clusterer = hdbscan.HDBSCAN(
            metric="cosine",
            algorithm="generic",
            min_cluster_size=2,
            min_samples=1,
            allow_single_cluster=True,
        )
        labels = clusterer.fit_predict(client_deltas_array)

        benign_indices = self._select_benign_indices(labels, num_clients)
        malicious_indices = list(set(range(num_clients)) - set(benign_indices))

        return malicious_indices, benign_indices, euclidean_distances

    def _select_benign_indices(self, labels: np.ndarray, num_clients: int) -> List[int]:
        """
        Pick the largest cluster as benign, with two false-positive guards:
          1. Fraction cap  — skip filtering if > max_filter_fraction would be removed.
          2. Ratio gate    — skip filtering if majority cluster is not clearly dominant.
        Noise points (-1) are kept with the benign majority by default.
        """
        non_noise_labels = np.unique(labels[labels != -1])

        if len(non_noise_labels) <= 1:
            return list(range(num_clients))

        counts = np.array([(labels == lbl).sum() for lbl in non_noise_labels])
        largest_idx   = np.argmax(counts)
        largest_label = non_noise_labels[largest_idx]
        largest_count = int(counts[largest_idx])
        noise_count   = int((labels == -1).sum())
        minority_count = num_clients - largest_count

        # Guard 1: don't filter too many at once
        if minority_count / num_clients > self.max_filter_fraction:
            print(
                f"  [FP guard] Would filter {minority_count}/{num_clients} "
                f"({minority_count/num_clients:.0%} > max {self.max_filter_fraction:.0%}). "
                f"Skipping filtering this round."
            )
            return list(range(num_clients))

        # Guard 2: majority cluster must clearly dominate
        second_largest  = int(np.delete(counts, largest_idx).max()) if len(counts) > 1 else 0
        challenger_size = max(second_largest, noise_count)
        if challenger_size > 0 and (largest_count / challenger_size) < self.min_cluster_ratio:
            print(
                f"  [FP guard] Cluster ratio {largest_count/challenger_size:.2f} "
                f"< min {self.min_cluster_ratio:.2f}. Clusters not distinct. "
                f"Skipping filtering this round."
            )
            return list(range(num_clients))

        return [i for i, lbl in enumerate(labels) if lbl == largest_label]

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        param_layers = [n for n in state_dict if "weight" in n or "bias" in n]
        return param_layers[-2:]