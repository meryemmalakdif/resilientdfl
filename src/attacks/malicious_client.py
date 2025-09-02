# src/attacks/malicious_client.py
from typing import Any, Dict, Optional, Sequence, Set
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.fl.baseclient import BenignClient


class PoisonedWrapper(Dataset):
    """
    Dataset wrapper that applies a trigger to selected indices and optionally forces label.
    Non-destructive: original dataset is not modified.
    Assumes base dataset __getitem__ returns (x, y).
    Trigger.apply must support the same input type returned by base dataset (PIL or Tensor).
    """
    def __init__(self, base_dataset: Dataset, poisoned_indices: Set[int], trigger, forced_label: Optional[int] = None, transform_after=None):
        self.base = base_dataset
        self.poisoned = set(poisoned_indices)
        self.trigger = trigger
        self.forced_label = forced_label
        # If you want to re-apply transforms after injecting trigger (when base returns PIL),
        # pass transform_after. Default tries to reuse base_dataset.transform if present.
        self.transform_after = transform_after if transform_after is not None else getattr(base_dataset, "transform", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.poisoned:
            x = self.trigger.apply(x)
            if self.forced_label is not None:
                y = self.forced_label
        # If we want to re-apply transforms to the possibly modified PIL, do it here.
        if self.transform_after is not None and (not hasattr(self.base, "transform") or self.base.transform is None):
            # Only apply transform_after when base didn't already transform input
            return self.transform_after(x), y
        return x, y


class MaliciousClient(BenignClient):
    """
    Malicious client that can (1) select local samples to poison, (2) apply a trigger,
    and (3) optionally use a poisoner module to craft the model update.

    Parameters (in addition to BenignClient args):
    - selector: object with select(indices, labels=None, num_poison=...) -> list_of_indices
    - trigger: object with apply(x) -> x_triggered
    - poisoner: optional object implementing poison_and_train(client, selector, trigger, **kwargs)
                If provided, MaliciousClient.local_train delegates to poisoner.poison_and_train.
    - target_label: label to assign to poisoned samples (if using naive poisoning)
    - poison_fraction: fraction of local samples to poison when using naive poisoning (ignored if poisoner is provided)
    """
    def __init__(
        self,
        id: int,
        trainloader: Optional[DataLoader],
        testloader: Optional[DataLoader],
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        epochs: int = 1,
        device: Optional[torch.device] = None,
        selector=None,
        trigger=None,
        poisoner=None,
        target_label: Optional[int] = None,
        poison_fraction: float = 0.1,
    ):
        super().__init__(
            id=id,
            trainloader=trainloader,
            testloader=testloader,
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            device=device,
        )
        self.selector = selector
        self.trigger = trigger
        self.poisoner = poisoner
        self.target_label = target_label
        self.poison_fraction = float(poison_fraction)

    # --------- small dataset helpers ----------
    @staticmethod
    def _get_labels_array(dataset) -> Optional[np.ndarray]:
        """
        Try common attributes (.targets, .labels, .samples) to gather labels array.
        If dataset is a Subset, try to map into underlying dataset.
        """
        # If Subset, resolve
        if isinstance(dataset, Subset):
            base = dataset.dataset
            idxs = np.array(dataset.indices)
            # attempt to extract base labels and index by idxs
            if hasattr(base, "targets"):
                return np.array(base.targets)[idxs]
            if hasattr(base, "labels"):
                return np.array(base.labels)[idxs]
            if hasattr(base, "samples"):
                return np.array([s[1] for s in np.array(base.samples)[idxs]])
            # fallback: iterate (slower)
            labels = []
            for i in range(len(dataset)):
                _, y = dataset[i]
                labels.append(int(y))
            return np.array(labels)
        else:
            base = dataset
            if hasattr(base, "targets"):
                return np.array(base.targets)
            if hasattr(base, "labels"):
                return np.array(base.labels)
            if hasattr(base, "samples"):
                return np.array([s[1] for s in base.samples])
            # fallback: iterate
            labels = []
            for i in range(len(base)):
                _, y = base[i]
                labels.append(int(y))
            return np.array(labels)

    def _select_poison_indices(self, num_poison: Optional[int] = None):
        """
        Run selector to pick indices to poison. Returns a set of indices relative to the client's dataset
        (i.e., indices in range(len(self.trainloader.dataset))).
        - If selector is None, chooses random indices using poison_fraction.
        """
        local_dataset = self.trainloader.dataset
        N = len(local_dataset)
        if N == 0:
            return set()

        # indices relative to local dataset (0..N-1)
        local_indices = list(range(N))

        # gather labels if available (relative)
        labels = self._get_labels_array(local_dataset)

        # determine num_poison
        if num_poison is None:
            if isinstance(self.poison_fraction, float) and self.poison_fraction <= 1.0:
                num_poison = max(1, int(round(self.poison_fraction * N)))
            else:
                num_poison = int(self.poison_fraction)

        if self.selector is None:
            # default: random uniform selection
            rng = np.random.RandomState(int(self.get_id()) + 0xC0FFEE)
            chosen = rng.choice(local_indices, size=min(num_poison, N), replace=False).tolist()
            return set(chosen)

        # else delegate to selector; the selector may expect global indices or relative indices depending on implementation.
        # We try to call selector.select with local indices and local labels (if available)
        try:
            chosen = self.selector.select(local_indices, labels=labels, num_poison=num_poison)
            # ensure chosen are ints and within range
            chosen = [int(x) for x in chosen]
            chosen = [x for x in chosen if 0 <= x < N]
            return set(chosen[:num_poison])
        except Exception:
            # fallback to random if selector fails
            rng = np.random.RandomState(int(self.get_id()) + 0xBEEF)
            chosen = rng.choice(local_indices, size=min(num_poison, N), replace=False).tolist()
            return set(chosen)

    # --------- core local_train override ----------
    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Main entry for local training. Behavior:
        - If self.poisoner is provided and has method poison_and_train(client, selector, trigger, **kwargs),
          delegate to it and return its result (it must return the same dict structure as BenignClient.local_train).
        - Otherwise perform naive poisoning: select indices, wrap dataset with PoisonedWrapper,
          swap self.trainloader, call BenignClient.local_train(...), restore loader, and return results.
        """
        # 1) If poisoner implements poison_and_train, delegate
        if self.poisoner is not None and hasattr(self.poisoner, "poison_and_train"):
            # poisoner is expected to handle training & return update dict
            return self.poisoner.poison_and_train(
                client=self,
                selector=self.selector,
                trigger=self.trigger,
                epochs=epochs,
                round_idx=round_idx,
                global_params=self.get_params()
            )

        # 2) naive poisoning: construct poisoned DataLoader and call parent's local_train
        if self.trainloader is None:
            # nothing to train on -> behave like benign
            return super().local_train(epochs, round_idx)

        # select which indices to poison (relative to local dataset)
        poisoned_rel = self._select_poison_indices()

        # map label forcing: if target_label is None, keep original labels (clean-label attack possible)
        forced_label = int(self.target_label) if self.target_label is not None else None

        # create poisoned dataset wrapper (preserve base dataset transforms if they exist)
        base_dataset = self.trainloader.dataset
        # If base_dataset is a Subset, its indices are already local; PoisonedWrapper expects indices relative to base dataset
        # (we selected local indices relative to the Subset already)
        poisoned_ds = PoisonedWrapper(base_dataset, poisoned_rel, trigger=self.trigger, forced_label=forced_label, transform_after=None)

        # match DataLoader params to original
        batch_size = getattr(self.trainloader, "batch_size", 32)
        shuffle = True  # for local training we usually shuffle poisoned data
        num_workers = getattr(self.trainloader, "num_workers", 0)
        poisoned_loader = DataLoader(poisoned_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        # swap loader, train using BenignClient.local_train (which will use self._model, optimizer, scheduler)
        orig_loader = self.trainloader
        try:
            self.trainloader = poisoned_loader
            result = super().local_train(epochs, round_idx)
        finally:
            # restore original loader even if training raises
            self.trainloader = orig_loader

        return result
