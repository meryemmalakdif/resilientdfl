from .utils import compute_distance, compute_weighted_average
from ..fl.baseserver import BaseServer, FedAvgAggregator
import torch
from typing import Dict, List, Optional


def multi_krum(updates: list[dict], num_byzantine: int = None, num_selected_updates: int = 1):
    num_updates = len(updates)
    if num_byzantine is None:
        num_byzantine = (num_updates - 2) // 2

    num_neighbors = max(1, num_updates - num_byzantine - 2)
    scores = []
    for i in range(num_updates):
        distances = []
        for j in range(num_updates):
            if i != j:
                distances.append(compute_distance(updates[i], updates[j]))
        distances.sort()
        score = sum(distances[:num_neighbors])
        scores.append({"id": i, "score": score})

    sorted_scores = sorted(scores, key=lambda x: x['score'])
    best_ids = [x["id"] for x in sorted_scores[:num_selected_updates]]
    return best_ids


class MKrumServer(FedAvgAggregator):
    def __init__(self, model, testloader=None, device=None, num_byzantine=None, num_selected_updates=1):
        super().__init__(model, testloader, device)
        self.num_byzantine = num_byzantine
        self.num_selected_updates = num_selected_updates

    def aggregate(self) -> Dict[str, torch.Tensor]:
        assert len(self.received_params) == len(self.received_lens) and len(self.received_params) > 0
        best_ids = multi_krum(self.received_params, self.num_byzantine, self.num_selected_updates)
        selected_updates = [self.received_params[idx] for idx in best_ids]
        selected_lens = [self.received_lens[idx] for idx in best_ids]
        aggregated_params = compute_weighted_average(selected_updates, selected_lens)

        # load averaged params to server model
        self.set_params({k: v.to(self.device) for k, v in aggregated_params.items()})

        self.received_params.clear()
        self.received_lens.clear()
        return aggregated_params
