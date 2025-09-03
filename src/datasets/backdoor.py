# src/datasets/backdoor.py
from typing import Optional, Sequence, List, Tuple
import random
from torch.utils.data import Dataset, DataLoader, Subset
import torch

class TriggeredTestset(Dataset):
    """
    Dataset wrapper used for backdoor evaluation: returns inputs with the trigger applied.
    - base_dataset: any torch Dataset (e.g., torchvision MNIST/CIFAR test dataset, or your adapter.dataset)
    - trigger: object with method apply(x) -> x_triggered. Should support input types returned by base_dataset:
               either PIL.Image or torch.Tensor.
    - keep_label: if True keep original labels (useful for clean-label attacks). If False and forced_label is not None,
                  relabel triggered samples to forced_label (classic BadNets evaluation).
    - forced_label: int or None. If not None and keep_label=False, the wrapper returns (triggered_input, forced_label).
    - fraction: float in (0,1] or int >=1:
         - if fraction <= 1.0 treated as fraction of dataset to trigger (randomly sample fraction*N examples)
         - if fraction >= 1 treated as absolute number of triggered examples (capped at dataset length)
    - seed: random seed used when selecting subset for partial triggering
    - transform_after: optional callable applied to the triggered input before returning (useful if base dataset
                       returned PIL and you want to re-apply transforms; default uses base_dataset.transform if present).
    """
    def __init__(
        self,
        base_dataset: Dataset,
        trigger,
        keep_label: bool = False,
        forced_label: Optional[int] = None,
        fraction: float = 1.0,
        seed: int = 0,
        transform_after = None,
    ):
        super().__init__()
        if fraction <= 0:
            raise ValueError("fraction must be > 0")
        self.base = base_dataset
        self.trigger = trigger
        self.keep_label = bool(keep_label)
        self.forced_label = None if keep_label else forced_label
        self.fraction = fraction
        self.seed = int(seed)
        # if transform_after is provided use it, otherwise use base.transform if available
        self.transform_after = transform_after if transform_after is not None else getattr(base_dataset, "transform", None)

        # prepare list of indices that will be triggered
        self._prepare_trigger_indices()

    def _prepare_trigger_indices(self):
        N = len(self.base)
        if self.fraction <= 1.0:
            n_trigger = max(1, int(round(self.fraction * N)))
        else:
            n_trigger = min(int(self.fraction), N)
        rng = random.Random(self.seed)
        all_idxs = list(range(N))
        rng.shuffle(all_idxs)
        self.triggered_indices = set(sorted(all_idxs[:n_trigger]))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        """
        Returns (x, y) where x is triggered if idx in triggered_indices, and y is either original label
        (if keep_label=True) or forced_label (if provided) else original label.
        """
        x, y = self.base[idx]
        # If base dataset had a transform that returns tensors, leaving transform_after None means
        # we will apply trigger on the returned type (tensor or PIL). If base returns PIL and transform_after
        # is provided, we will apply transform_after after triggering.
        if idx in self.triggered_indices:
            x = self.trigger.apply(x)
            if not self.keep_label and self.forced_label is not None:
                y = self.forced_label
        # optionally reapply transform (only useful if base returned PIL but you want tensors)
        if getattr(self.base, "transform", None) is None and self.transform_after is not None:
            x = self.transform_after(x)
        return x, y


def make_triggered_loader(
    base_dataset: Dataset,
    trigger,
    keep_label: bool = False,
    forced_label: Optional[int] = None,
    fraction: float = 1.0,
    seed: int = 0,
    transform_after = None,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    """
    Convenience helper: returns a DataLoader wrapping a TriggeredTestset.
    - base_dataset: dataset to wrap (e.g., adapter.dataset or test_loader.dataset)
    - trigger: trigger object with apply(x)
    - keep_label / forced_label / fraction: passed to TriggeredTestset
    - transform_after: optional transform callable to apply after triggering (e.g., transforms.ToTensor + Normalize)
    - batch_size / shuffle / num_workers: DataLoader args
    """
    ds = TriggeredTestset(
        base_dataset=base_dataset,
        trigger=trigger,
        keep_label=keep_label,
        forced_label=forced_label,
        fraction=fraction,
        seed=seed,
        transform_after=transform_after
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
