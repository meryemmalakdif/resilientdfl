from utils import NoiseDataset
from ..fl.baseserver import BaseServer, FedAvgAggregator
import torch
from typing import Dict, List, Optional, Tuple
import numpy as np
import hdbscan
from const import NUM_CLASSES, IMG_SIZE
import copy


class FlameServer(FedAvgAggregator):
    def __init__(self, model: torch.nn.Module, testloader=None, device: Optional[torch.device]=None, alpha=0.1):
        super().__init__(model, testloader, device)
        

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """Get names of last two layers."""
        layer_names = list(state_dict.keys())
        return layer_names[-2:]
    
    def detect_anomalies(self) -> List[int]:
        """
        Detect anomalies using deepsight defense.
        Returns:
            List[int]: Indices of detected anomalous updates.
        """
        pass

    def aggregate(self) -> Dict[str, torch.Tensor]:
        pass

    def _calculate_neups(self, local_model_updates: List[Dict[str, torch.Tensor]], num_classes: int, last_layer_name: str) -> Tuple[List[float], List[float], List[float]]:
        NEUPs, TEs, euclidean_distances = [], [], []

        last_layer_weight_name = last_layer_name + ".weight"
        last_layer_bias_name = last_layer_name + ".bias"

        # Calculate update norms and NEUPs
        for local_model_update in local_model_updates:
            # Calculate Euclidean distance
            flat_update = []
            for name, param in local_model_update.items():
                if 'weight' in name or 'bias' in name:
                    diff = param - self.global_model_params[name]
                    flat_update.append(diff.flatten())  # Keep as torch.Tensor
            euclidean_distances.append(torch.linalg.norm(torch.cat(flat_update)))

            # Calculate NEUPs
            diff_weight = torch.sum(torch.abs(local_model_update[last_layer_weight_name] - self.global_model_params[last_layer_weight_name]), dim=1) # weight
            diff_bias = torch.abs(local_model_update[last_layer_bias_name] - self.global_model_params[last_layer_bias_name]) # bias

            UPs_squared = (diff_bias + diff_weight) ** 2
            NEUP = UPs_squared / torch.sum(UPs_squared)

            NEUP_np = NEUP.cpu().numpy()
            NEUPs.append(NEUP_np)

            # Calculate TE
            max_NEUP = np.max(NEUP_np)
            threshold = (1 / num_classes) * max_NEUP
            TE = sum(1 for j in NEUP_np if j >= threshold)
            TEs.append(TE)

        NEUPs = np.reshape(NEUPs, (len(local_model_updates), num_classes))
        return NEUPs, TEs, euclidean_distances

    def _calculate_ddifs(self, local_model_updates: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        """Calculate DDifs using random noise inputs."""
        num_classes = NUM_CLASSES[self.config.dataset.upper()]
        img_height, img_width, num_channels = IMG_SIZE[self.config.dataset.upper()]

        self.global_model.eval()
        local_model = copy.deepcopy(self.global_model)
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
                        output_global = self.global_model(inputs)

                    # Division and summation (preserving your optimization)
                    ratio = torch.div(output_local, output_global + 1e-30)
                    DDif.add_(ratio.sum(dim=0))

                DDif /= self.num_samples
                seed_ddifs.append(DDif.cpu().numpy())

            DDifs.append(seed_ddifs)

        DDifs = np.reshape(DDifs, (self.num_seeds, len(local_model_updates), num_classes))
        return DDifs

    def _calculate_cosine_distances(self, local_model_updates: List[Dict[str, torch.Tensor]], last_layer_name) -> np.ndarray:
        """Calculate cosine distances between client updates."""
        N = len(local_model_updates)
        distances = np.zeros((N, N))

        # Get last layer parameters
        bias_name = last_layer_name + ".bias"

        for i in range(N):
            for j in range(i + 1, N):
                # Get bias differences
                bias_i = local_model_updates[i][bias_name] - self.global_model_params[bias_name].to(self.device)
                bias_j = local_model_updates[j][bias_name] - self.global_model_params[bias_name].to(self.device)

                # Calculate cosine distance using PyTorch (preserving your optimization)
                bias_i_flat = bias_i.flatten()
                bias_j_flat = bias_j.flatten()

                dot_product = torch.dot(bias_i_flat, bias_j_flat)
                norm_i = torch.linalg.norm(bias_i_flat)
                norm_j = torch.linalg.norm(bias_j_flat)

                similarity = dot_product / (norm_i * norm_j + 1e-10)
                dist = 1.0 - similarity.item()

                distances[i, j] = distances[j, i] = dist

        return distances

    def _dists_from_clust(self, clusters: np.ndarray, N: int) -> np.ndarray:
        """Calculate distance matrix from cluster assignments (following reference)."""
        pairwise_dists = np.ones((N, N))
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if clusters[i] == clusters[j] and clusters[i] != -1:
                    pairwise_dists[i][j] = 0


        return pairwise_dists