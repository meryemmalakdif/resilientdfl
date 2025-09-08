import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy

import hdbscan


from ..fl.baseserver import FedAvgAggregator
from .utils import NoiseDataset # Assuming you have this helper from your project structure
from .const import NUM_CLASSES, IMG_SIZE # Assuming these constants are defined


class DeepSightServer(FedAvgAggregator):
    """
    Implements the DeepSight defense mechanism.

    This server aggregates client updates after filtering out malicious clients
    detected by a multi-metric clustering approach.
    """
    def __init__(self, model: torch.nn.Module, testloader: DataLoader = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)
        
        if hdbscan is None:
            raise ImportError("hdbscan is not installed. Please install it to use DeepSightServer.")
            
        self.config = config if config is not None else {}
        
        # Deepsight-specific parameters from the config
        self.num_samples = self.config.get('deepsight_num_samples', 256)
        self.num_seeds = self.config.get('deepsight_num_seeds', 3)
        self.deepsight_batch_size = self.config.get('deepsight_batch_size', 64)
        # --- MODIFICATION: Added tau threshold from reference ---
        self.deepsight_tau = self.config.get('deepsight_tau', 0.5) # Default 0.5, paper uses 1/3
        
        self.global_model_params = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Filters malicious updates using DeepSight and then performs robust aggregation.
        """
        if not self.updates:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        # --- MODIFICATION: Get euclidean distances along with anomalies for clipping ---
        anomalous_clients_indices, euclidean_distances = self.detect_anomalies()
        
        print(f"Deepsight detected {len(anomalous_clients_indices)} anomalous clients.")
        if anomalous_clients_indices:
            print(f"Filtering out clients at indices: {anomalous_clients_indices}")

        benign_updates = [update for i, update in enumerate(self.updates) if i not in anomalous_clients_indices]
        
        if not benign_updates:
            print("Warning: Deepsight filtered out all clients. Global model will not be updated.")
            self.updates.clear()
            return self.get_params()

        # --- MODIFICATION: Robust aggregation with clipping (from reference) ---
        benign_indices = [i for i, update in enumerate(self.updates) if i not in anomalous_clients_indices]
        benign_distances = [euclidean_distances[i] for i in benign_indices]
        clip_norm = torch.median(torch.tensor(benign_distances)).item()

        total_samples = sum(update['length'] for update in benign_updates)
        avg_weights = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}

        for k in avg_weights.keys():
            # Get the global param for calculating the diff
            global_param = self.global_model_params[k].to(self.device)
            for i, update in enumerate(benign_updates):
                # Get the original index to find the correct euclidean distance
                original_index = benign_indices[i]
                diff = update['weights'][k].to(self.device) - global_param
                
                # Apply clipping
                if euclidean_distances[original_index] > clip_norm:
                    diff *= clip_norm / euclidean_distances[original_index]
                
                weight_fraction = update['length'] / total_samples
                avg_weights[k] += diff * weight_fraction
        
        # Apply the aggregated update to the global model
        final_state_dict = self.model.state_dict()
        for k in avg_weights.keys():
            final_state_dict[k] += avg_weights[k]
        self.model.load_state_dict(final_state_dict)
        
        self.updates.clear()
        self.global_model_params = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        return self.get_params()

    def detect_anomalies(self) -> Tuple[List[int], List[float]]:
        """
        Orchestrates the DeepSight detection process.
        Returns a list of malicious indices and a list of all euclidean distances.
        """
        num_clients = len(self.updates)
        if num_clients < 2:
            return [], []

        local_model_updates = [u['weights'] for u in self.updates]
        last_layer_name = self._get_last_layers(local_model_updates[0])[-1].split('.')[0]
        num_classes = self.model.state_dict()[f"{last_layer_name}.weight"].shape[0]

        # --- Metric 1: NEUP-based distance ---
        neups, TEs, euclidean_distances = self._calculate_neups(local_model_updates, num_classes, last_layer_name)
        
        # --- MODIFICATION: More sophisticated detection from reference ---
        # Get initial labels based on Threshold Exceedings (TEs)
        classification_boundary = np.median(TEs) if TEs else 0
        # Label is True if malicious (low TE), False if benign (high TE)
        te_labels = [te <= classification_boundary * 0.5 for te in TEs]

        # --- Metric 2: DDif-based distance ---
        ddifs_per_seed = self._calculate_ddifs(local_model_updates)
        
        # --- Metric 3: Cosine distance ---
        dist_cosine = self._calculate_cosine_distances(local_model_updates, last_layer_name)

        # --- Clustering and merging ---
        neup_clusters = hdbscan.HDBSCAN().fit_predict(neups)
        neup_dists = self._dists_from_clust(neup_clusters, num_clients)

        cosine_clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(dist_cosine)
        cosine_dists = self._dists_from_clust(cosine_clusters, num_clients)
        
        ddif_dists_list = []
        for i in range(self.num_seeds):
            ddif_clusters = hdbscan.HDBSCAN().fit_predict(ddifs_per_seed[i])
            ddif_dists_list.append(self._dists_from_clust(ddif_clusters, num_clients))

        merged_ddif_dists = np.average(ddif_dists_list, axis=0)
        
        merged_distances = np.mean([merged_ddif_dists, neup_dists, cosine_dists], axis=0)
        final_clusters = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2, allow_single_cluster=True).fit_predict(merged_distances)

        # --- Classify clusters to find malicious clients ---
        benign_clients_set = set()
        malicious_clients_set = set()

        unique_clusters = [l for l in np.unique(final_clusters) if l != -1]
        
        for cluster_label in unique_clusters:
            member_indices = np.where(final_clusters == cluster_label)[0]
            # Count benign clients (where te_label is False)
            num_benign = sum(1 for i in member_indices if not te_labels[i])
            
            if len(member_indices) > 0 and (num_benign / len(member_indices)) >= self.deepsight_tau:
                benign_clients_set.update(member_indices)
            else:
                malicious_clients_set.update(member_indices)
        
        # Handle outliers (label -1) based on their TE label
        outlier_indices = np.where(final_clusters == -1)[0]
        for i in outlier_indices:
            if not te_labels[i]: # Benign outlier
                benign_clients_set.add(i)
            else: # Malicious outlier
                malicious_clients_set.add(i)

        return list(malicious_clients_set), euclidean_distances
    
    def _dists_from_clust(self, clusters: np.ndarray, N: int) -> np.ndarray:
        """Calculate distance matrix from cluster assignments."""
        pairwise_dists = np.ones((N, N))
        for i in range(N):
            for j in range(i, N):
                if clusters[i] == clusters[j] and clusters[i] != -1:
                    pairwise_dists[i, j] = pairwise_dists[j, i] = 0
        return pairwise_dists
    
    def _get_largest_cluster(self, clusterer: hdbscan.HDBSCAN) -> set:
        """Finds the set of indices belonging to the largest cluster."""
        labels = clusterer.labels_
        if len(labels) == 0: return set()
        unique_labels = [l for l in np.unique(labels) if l != -1]
        if not unique_labels: return set()
        largest_cluster_label = max(unique_labels, key=lambda l: np.sum(labels == l))
        return set(np.where(labels == largest_cluster_label)[0])

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """Get names of last two layers."""
        layer_names = list(state_dict.keys())
        param_layers = [name for name in layer_names if 'weight' in name or 'bias' in name]
        return param_layers[-2:]

    def _calculate_neups(self, local_model_updates: List[Dict[str, torch.Tensor]], num_classes: int, last_layer_name: str) -> Tuple[np.ndarray, List[float], List[float]]:
        NEUPs, TEs, euclidean_distances = [], [], []
        
        last_layer_weight_name = last_layer_name + ".weight"
        last_layer_bias_name = last_layer_name + ".bias"

        for local_model_update in local_model_updates:
            device = local_model_update[last_layer_weight_name].device
            
            # --- MODIFICATION: Calculate Euclidean distance for clipping ---
            flat_update = []
            for name, param in local_model_update.items():
                if 'weight' in name or 'bias' in name:
                    diff = param.to(device) - self.global_model_params[name].to(device)
                    flat_update.append(diff.flatten())
            euclidean_distances.append(torch.linalg.norm(torch.cat(flat_update)).item())

            global_weight = self.global_model_params[last_layer_weight_name].to(device)
            global_bias = self.global_model_params[last_layer_bias_name].to(device)

            diff_weight = torch.sum(torch.abs(local_model_update[last_layer_weight_name] - global_weight), dim=1)
            diff_bias = torch.abs(local_model_update[last_layer_bias_name] - global_bias)

            UPs_squared = (diff_bias + diff_weight) ** 2
            NEUP = UPs_squared / (torch.sum(UPs_squared) + 1e-10)
            
            NEUP_np = NEUP.cpu().numpy()
            NEUPs.append(NEUP_np)

            # --- MODIFICATION: Calculate Threshold Exceedings (TEs) ---
            max_NEUP = np.max(NEUP_np)
            threshold = (1 / num_classes) * max_NEUP if num_classes > 0 else 0
            TE = np.sum(NEUP_np >= threshold)
            TEs.append(TE)

        return np.array(NEUPs), TEs, euclidean_distances

    def _calculate_ddifs(self, local_model_updates: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        """Calculate DDifs using random noise inputs."""
        dataset_name = self.config.get('dataset', '').upper()
        if not dataset_name or dataset_name not in NUM_CLASSES:
             raise ValueError("Dataset name must be provided in config for DeepSight")

        num_classes = NUM_CLASSES[dataset_name]
        img_height, img_width, num_channels = IMG_SIZE[dataset_name]

        self.model.eval()
        local_model = copy.deepcopy(self.model)
        DDifs = []
        for seed in range(self.num_seeds):
            torch.manual_seed(seed)
            dataset = NoiseDataset((num_channels, img_height, img_width), self.num_samples)
            loader = torch.utils.data.DataLoader(dataset, self.deepsight_batch_size, shuffle=False)

            seed_ddifs = []
            for local_update in local_model_updates:
                local_model.load_state_dict(local_update)
                local_model.eval()

                DDif = torch.zeros(num_classes, device=self.device)
                for inputs in loader:
                    inputs = inputs.to(self.device)
                    with torch.no_grad():
                        output_local = local_model(inputs)
                        output_global = self.model(inputs)

                    ratio = torch.div(output_local, output_global + 1e-30)
                    DDif.add_(ratio.sum(dim=0))

                DDif /= self.num_samples
                seed_ddifs.append(DDif.cpu().numpy())
            DDifs.append(seed_ddifs)
        return np.array(DDifs)

    def _calculate_cosine_distances(self, local_model_updates: List[Dict[str, torch.Tensor]], last_layer_name: str) -> np.ndarray:
        """Calculate cosine distances between client updates."""
        N = len(local_model_updates)
        distances = np.zeros((N, N))
        bias_name = last_layer_name + ".bias"
        
        bias_diffs = []
        for i in range(N):
            device = local_model_updates[i][bias_name].device
            global_bias = self.global_model_params[bias_name].to(device)
            bias_diffs.append((local_model_updates[i][bias_name] - global_bias).flatten())

        for i in range(N):
            for j in range(i + 1, N):
                bias_i_flat = bias_diffs[i]
                bias_j_flat = bias_diffs[j]
                similarity = F.cosine_similarity(bias_i_flat, bias_j_flat, dim=0, eps=1e-10)
                dist = 1.0 - similarity.item()
                distances[i, j] = distances[j, i] = dist
        return distances

