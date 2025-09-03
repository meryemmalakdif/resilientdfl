from utils import compute_distance, compute_weighted_average
from ..fl.baseserver import BaseServer, FedAvgAggregator
import torch
from typing import Dict, List, Optional


def multi_krum(updates: list[dict], num_byzantine: int = None, num_selected_updates: int =1):
    """
    Implementation of the Multi-Krum defense, for each update w_i a score s(i) = sum_j( ||w_i - w_j||² ),
    where j runs over (len(updates) - num_byzantine - 2). Then num_selected_updates with the least scores
    are selected.
    Args:
        updates (list[dict]): the list of local updates.
        num_byzantine (int): assumed number of byzantine nodes (f in the krum paper).
        num_selected_updates (int): the number of updates selected for aggregation, default to 1, which is the Krum defense.
    Returns:
        list[dict]: a list of updates elected for aggregation.
    """
    
    num_updates = len(updates)
    if num_byzantine == None:
        num_byzantine = (num_updates - 2) // 2
    
    num_neighbors = num_updates - num_byzantine - 2
    scores = []
    for i in range(num_updates):
        distances = []
        for j in range(num_updates):
            if i!=j:
                distances.append(compute_distance(updates[i], updates[j]))
        distances.sort()
        score = sum(distances[:num_neighbors])
        scores.append({"id": i, "score":score})

    sorted_scores = sorted(scores, key=lambda x: x['score'])
    
    best_ids = [x["id"] for x in sorted_scores[:num_selected_updates]]
    elected_updates = [updates[idx] for idx in best_ids]

    return elected_updates   


class MKrumServer(FedAvgAggregator):
    def __init__(self, model: torch.nn.Module, testloader=None, device: Optional[torch.device]=None, num_byzantine: int = None, num_selected_updates: int =1):
        super().__init__(model, testloader, device)
        self.num_byzantine = num_byzantine
        self.num_selected_updates = num_selected_updates

    def aggregate(self) -> Dict[str, torch.Tensor]:
        assert len(self.received_params) == len(self.received_lens) and len(self.received_params) > 0
        selected_updates = multi_krum(self.received_params, self.num_byzantine, self.num_selected_updates)
        aggregated_params = compute_weighted_average(selected_updates)
        self.received_params = []
        self.received_lens = []
        return aggregated_params