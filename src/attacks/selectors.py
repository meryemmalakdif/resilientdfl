from abc import ABC, abstractmethod
from typing import Sequence, List
import numpy as np

class SampleSelector(ABC):
    @abstractmethod
    def select(self, indices: Sequence[int], labels=None, num_poison: int = 1) -> List[int]:
        """Return a subset of indices to poison (from the provided list)."""

class RandomSelector(SampleSelector):
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
    def select(self, indices, labels=None, num_poison: int = 1):
        idxs = list(indices)
        if num_poison >= len(idxs):
            return idxs
        return list(self.rng.choice(idxs, size=num_poison, replace=False))

class ClassBalancedSelector(SampleSelector):
    """Pick samples from classes the attacker prefers (e.g., pick minority labels)."""
    def __init__(self, target_label=None, seed: int = 0):
        self.target_label = target_label
        self.rng = np.random.RandomState(seed)
    def select(self, indices, labels=None, num_poison: int = 1):
        if labels is None:
            return RandomSelector(self.rng.randint(0,1e9)).select(indices, None, num_poison)
        # prefer indices where labels == target_label if provided
        idxs = np.array(indices)
        lbls = np.array(labels)[idxs]
        if self.target_label is None:
            # choose class with fewest samples on this client
            uniq, counts = np.unique(lbls, return_counts=True)
            self.target_label = int(uniq[np.argmin(counts)])
        cand = idxs[lbls == self.target_label]
        if len(cand) == 0:
            return RandomSelector(self.rng.randint(0,1e9)).select(indices, None, num_poison)
        k = min(len(cand), num_poison)
        return list(self.rng.choice(cand, size=k, replace=False))

# TSSO: Efficient and persistent backdoor attack by boundary trigger set constructing against federated learning.
# boundary-based sample selection: get latent representations of all samples, use LSH to find samples near decision boundary
class BoundaryBasedSelector(SampleSelector):
    def __init__(self):
        pass

    def select(self, indices, labels=None, num_poison: int = 1):
        pass  # Placeholder for actual implementation


