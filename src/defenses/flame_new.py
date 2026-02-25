


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np



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


class RobustFlameServer(FedAvgAggregator):
    """
    FLAME + Structural False Positive Rescue (S-FPR).

    Pipeline:
        Stage 1 — FLAME (HDBSCAN cosine clustering)  [unchanged]
                  Flags geometrically suspicious clients.
                  Problem: also flags non-IID benign clients (~45% FP rate).

        Stage 2 — Structural Deviation Analysis on the flagged set only.
                  Three signals, each measuring a different property of
                  Δj = gj − benign_cluster_average:

                  Signal 1 │ Sparsity Score  (ℓ2 / ℓ1 of Δj)
                  ─────────┼──────────────────────────────────────────────
                            │ Backdoor updates optimize for a specific trigger
                            │ response → concentrate energy in few dimensions.
                            │ Benign non-IID updates reflect data heterogeneity
                            │ → spread deviation broadly across dimensions.
                            │ Higher ratio = more concentrated = more suspicious.

                  Signal 2 │ Spectral Concentration  (PCA of Δj)
                  ─────────┼──────────────────────────────────────────────
                            │ Across the flagged group, compute the top-k
                            │ principal components of {Δj}.
                            │ Backdoor client: deviation aligns strongly with
                            │ 1–2 dominant directions (trigger subspace).
                            │ Benign minority client: variance is diffuse.
                            │ Score = fraction of variance explained by top-k PCs.

                  Signal 3 │ Last-Layer Energy Ratio  (‖Δj_last‖ / ‖Δj_all‖)
                  ─────────┼──────────────────────────────────────────────
                            │ Backdoors embed trigger→class mapping in the
                            │ final classification layer.
                            │ Benign clients modify all layers proportionally.
                            │ High last-layer concentration = suspicious.

                  Decision:
                    All three scores are z-scored within the flagged group
                    (relative to peers, not a fixed global threshold).
                    Client is confirmed malicious if ≥ min_signals scores
                    exceed z_thresh.  Otherwise → false positive → reinstate.
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
        self.lamda       = self.config.get("flame_lamda", 0.001)
        self.eta         = self.config.get("flame_eta", 1.0)
        self.pca_top_k   = self.config.get("flame_pca_top_k", 3)   # PCA components
        self.z_thresh    = self.config.get("flame_z_thresh", 1.5)   # outlier z-score
        self.min_signals = self.config.get("flame_min_signals", 2)  # votes to confirm

        print(
            f"Initialized FlameServer (+S-FPR) | "
            f"lamda={self.lamda} eta={self.eta} "
            f"pca_top_k={self.pca_top_k} z_thresh={self.z_thresh} "
            f"min_signals={self.min_signals}"
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> Dict[str, torch.Tensor]:
        if not self.received_params:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        n = len(self.received_params)

        # ── Stage 1: original FLAME clustering ───────────────────────────
        malicious_indices, benign_indices, euclidean_distances = self.detect_anomalies()
        print(f"  [FLAME  Stage 1] Flagged {len(malicious_indices)}/{n}: {malicious_indices}")

        # ── Stage 2: structural rescue ────────────────────────────────────
        if malicious_indices and benign_indices:
            rescued, confirmed = self._structural_rescue(
                malicious_indices, benign_indices
            )
            if rescued:
                print(f"  [S-FPR  Stage 2] Reinstated (false positives) : {rescued}")
            if confirmed:
                print(f"  [S-FPR  Stage 2] Confirmed malicious           : {confirmed}")

            benign_indices    = sorted(benign_indices + rescued)
            malicious_indices = confirmed
        else:
            print("  [S-FPR  Stage 2] Nothing to re-examine.")

        print(f"  [FLAME  Final  ] Aggregating {len(benign_indices)}/{n} clients.")

        if not benign_indices:
            print("Warning: All clients filtered. Model not updated.")
            self.received_params.clear()
            self.received_lens.clear()
            return self.get_params()

        # ── Aggregation with clipping + noise (original FLAME) ───────────
        benign_distances = [euclidean_distances[i] for i in benign_indices]
        clip_norm = (
            torch.median(torch.tensor(benign_distances)).item()
            if benign_distances else 1.0
        )

        weight_accumulator = {
            name: torch.zeros_like(param)
            for name, param in self.model.state_dict().items()
        }
        global_params = self.get_params()

        for idx in benign_indices:
            local_params = self.received_params[idx]
            weight = 1.0 / len(benign_indices)
            for name, param in local_params.items():
                if name.endswith("num_batches_tracked"):
                    continue
                diff = param.to(self.device) - global_params[name].to(self.device)
                if euclidean_distances[idx] > clip_norm:
                    diff *= clip_norm / euclidean_distances[idx]
                weight_accumulator[name].add_(diff * weight)

        final_state_dict = self.model.state_dict()
        for name, param in final_state_dict.items():
            if name in weight_accumulator:
                param.add_(weight_accumulator[name] * self.eta)
                if "weight" in name or "bias" in name:
                    std_dev = self.lamda * clip_norm
                    noise   = torch.normal(0, std_dev, param.shape, device=param.device)
                    param.add_(noise)

        self.model.load_state_dict(final_state_dict)
        self.received_params.clear()
        self.received_lens.clear()
        return self.get_params()

    # ------------------------------------------------------------------
    # Stage 2: Structural False Positive Rescue
    # ------------------------------------------------------------------

    def _structural_rescue(
        self,
        malicious_indices: List[int],
        benign_indices: List[int],
    ) -> Tuple[List[int], List[int]]:
        """
        Re-examine each FLAME-flagged client using three structural signals.
        Decision is via majority vote — a client must trigger >= min_signals
        to be confirmed malicious; otherwise it is reinstated as a false positive.
        """
        global_params    = self.get_params()
        last_layer_names = self._get_last_layers(self.received_params[0])

        # ── Helper: get full and last-layer deviation vectors ─────────────
        def get_deviation(
            local_params: Dict,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            full_parts, last_parts = [], []
            for name, param in local_params.items():
                if name.endswith("num_batches_tracked"):
                    continue
                if "weight" not in name and "bias" not in name:
                    continue
                diff = param.to(self.device) - global_params[name].to(self.device)
                full_parts.append(diff.flatten())
                if name in last_layer_names:
                    last_parts.append(diff.flatten())
            full_vec = torch.cat(full_parts)
            last_vec = (
                torch.cat(last_parts)
                if last_parts
                else torch.zeros(1, device=self.device)
            )
            return full_vec, last_vec

        # Benign cluster average — reference direction for Δj
        benign_full_vecs = [
            get_deviation(self.received_params[i])[0] for i in benign_indices
        ]
        benign_avg = torch.stack(benign_full_vecs).mean(dim=0)  # shape [d]

        # ── Compute raw scores for each flagged client ────────────────────
        sparsity_scores   = []
        last_layer_scores = []
        delta_np_list     = []   # for batch PCA

        for idx in malicious_indices:
            full_vec, last_vec = get_deviation(self.received_params[idx])

            # Δj = deviation relative to benign cluster average
            delta = full_vec - benign_avg

            # ── Signal 1: Sparsity (ℓ2 / ℓ1) ────────────────────────────
            l2 = torch.linalg.norm(delta).item()
            l1 = delta.abs().sum().item()
            sparsity_scores.append(l2 / (l1 + 1e-10))

            # ── Signal 3: Last-layer energy ratio ────────────────────────
            # Use the last-layer slice of the full deviation, not last_vec
            # (last_vec is relative to global, delta is relative to benign avg)
            last_layer_full = get_deviation(self.received_params[idx])[1]
            last_layer_delta = last_layer_full - benign_avg[-last_layer_full.shape[0]:]
            last_norm = torch.linalg.norm(last_layer_delta).item()
            last_layer_scores.append(last_norm / (l2 + 1e-10))

            delta_np_list.append(delta.cpu().float().numpy())

        # ── Signal 2: Spectral concentration (batch PCA) ─────────────────
        spectral_scores = self._compute_spectral_scores(delta_np_list)

        # ── Z-score all signals within the flagged group ──────────────────
        def z_score(vals: List[float]) -> List[float]:
            arr  = np.array(vals, dtype=np.float64)
            std  = arr.std() + 1e-10
            return list((arr - arr.mean()) / std)

        z_sparsity   = z_score(sparsity_scores)
        z_spectral   = z_score(spectral_scores)
        z_last_layer = z_score(last_layer_scores)

        # ── Majority-vote decision ────────────────────────────────────────
        rescued, confirmed = [], []

        for i, idx in enumerate(malicious_indices):
            votes = (
                int(z_sparsity[i]   > self.z_thresh)
                + int(z_spectral[i]   > self.z_thresh)
                + int(z_last_layer[i] > self.z_thresh)
            )
            verdict = "MALICIOUS" if votes >= self.min_signals else "FALSE POS"
            print(
                f"    client {idx:3d} │ "
                f"sparsity_z={z_sparsity[i]:+.2f}  "
                f"spectral_z={z_spectral[i]:+.2f}  "
                f"last_layer_z={z_last_layer[i]:+.2f}  "
                f"votes={votes}/{self.min_signals} → {verdict}"
            )
            if votes >= self.min_signals:
                confirmed.append(idx)
            else:
                rescued.append(idx)

        return rescued, confirmed

    # ------------------------------------------------------------------
    # Signal 2 helper: spectral concentration via SVD
    # ------------------------------------------------------------------

    def _compute_spectral_scores(
        self, delta_vecs: List[np.ndarray]
    ) -> List[float]:
        """
        Project each flagged client's deviation onto the top-k principal
        components of the flagged group.  Score = fraction of the client's
        own variance captured by those top-k directions.

        Backdoor client: its deviation lies mostly in the trigger subspace
        which dominates the top PCs → high score.
        Benign minority client: deviation is diffuse → low score.
        """
        if len(delta_vecs) < 2:
            return [0.0] * len(delta_vecs)

        try:
            mat = np.stack(delta_vecs, axis=0).astype(np.float32)  # [n, d]
            k   = min(self.pca_top_k, mat.shape[0] - 1, mat.shape[1])
            if k < 1:
                return [0.0] * len(delta_vecs)

            mat_c = mat - mat.mean(axis=0, keepdims=True)           # center
            _, _, Vt = np.linalg.svd(mat_c, full_matrices=False)    # SVD
            top_Vt = Vt[:k]                                          # [k, d]

            scores = []
            for vec in mat_c:
                total_var    = float(np.dot(vec, vec)) + 1e-10
                projected    = top_Vt @ vec                          # [k]
                captured_var = float(np.dot(projected, projected))
                scores.append(captured_var / total_var)

            return scores

        except Exception as e:
            print(f"  [S-FPR] SVD failed ({e}), zeroing spectral scores.")
            return [0.0] * len(delta_vecs)

    # ------------------------------------------------------------------
    # Original FLAME clustering (completely unchanged)
    # ------------------------------------------------------------------

    def detect_anomalies(self) -> Tuple[List[int], List[int], List[float]]:
        num_clients = len(self.received_params)
        if num_clients < 2:
            return [], list(range(num_clients)), []

        last_layer_names   = self._get_last_layers(self.received_params[0])
        all_client_weights = []
        euclidean_distances = []
        global_params_cpu  = self.get_params()

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

        all_indices       = set(range(num_clients))
        malicious_indices = list(all_indices - set(benign_indices))
        return malicious_indices, benign_indices, euclidean_distances

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        layer_names  = list(state_dict.keys())
        param_layers = [n for n in layer_names if "weight" in n or "bias" in n]
        return param_layers[-2:]