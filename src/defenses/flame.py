from utils import compute_distance, compute_weighted_average
from ..fl.baseserver import BaseServer, FedAvgAggregator
import torch
from typing import Dict, List, Optional

import hdbscan

class FlameServer(FedAvgAggregator):
    def __init__(self, model: torch.nn.Module, testloader=None, device: Optional[torch.device]=None, alpha=0.1):
        super().__init__(model, testloader, device)
        self.alpha = alpha

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """Get names of last two layers."""
        layer_names = list(state_dict.keys())
        return layer_names[-2:]
    
    def detect_anomalies(self) -> List[int]:
        """
        Detect anomalies using FLAME defense.
        Returns:
            List[int]: Indices of detected anomalous updates.
        """
        pass

    def aggregate(self) -> Dict[str, torch.Tensor]:
        pass