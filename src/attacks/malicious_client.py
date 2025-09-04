from typing import Any, Dict, Optional, Sequence, Set, List
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.fl.baseclient import BenignClient

logger = logging.getLogger(__name__)


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
        # If base didn't already apply transforms but we have a transform_after, apply it.
        if getattr(self.base, "transform", None) is None and self.transform_after is not None:
            return self.transform_after(x), y
        return x, y


# ----------------- MaliciousClient -----------------
class MaliciousClient(BenignClient):
    """
    Malicious client composed of (selector, trigger, poisoner).
    - selector: picks indices to poison (relative indices 0..N-1)
    - trigger: Trigger.apply(x) to embed trigger
    - poisoner: object with poison_and_train(client, selected_indices, trigger, global_params, epochs, round_idx, **kwargs)
                If omitted, we fall back to naive poisoning (PoisonedWrapper + BenignClient.local_train).
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

    # ---------- helpers to get labels ----------
    @staticmethod
    def _get_labels_array(dataset) -> Optional[np.ndarray]:
        """
        Attempt to extract labels for the given dataset or Subset.
        Returns labels aligned with dataset indices (length == len(dataset)).
        """
        if isinstance(dataset, Subset):
            base = dataset.dataset
            idxs = np.array(dataset.indices)
            # try common attributes on base and index into them
            if hasattr(base, "targets"):
                return np.array(base.targets)[idxs]
            if hasattr(base, "labels"):
                return np.array(base.labels)[idxs]
            if hasattr(base, "samples"):
                return np.array([s[1] for s in np.array(base.samples)[idxs]])
            # fallback iterate
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
            labels = []
            for i in range(len(base)):
                _, y = base[i]
                labels.append(int(y))
            return np.array(labels)

    # ---------- flexible selector caller ----------
    def _call_selector(self, selector, dataset, local_indices: Sequence[int], num_poison: int, labels: Optional[np.ndarray]) -> List[int]:
        """
        Try several selector.select signatures to maximize compatibility with different selector implementations.
        Preferred call: selector.select(dataset, indices, num_poison, labels=labels)
        Fallbacks:
          - selector.select(indices, labels=labels, num_poison=num_poison)
          - selector.select(indices, num_poison)
          - selector.select(indices)
        Returns list of chosen indices relative to local dataset (0..N-1).
        """
        if selector is None:
            return []
        # try preferred signature first
        try:
            return list(selector.select(dataset, local_indices, int(num_poison), labels=labels))
        except TypeError:
            pass
        except Exception as e:
            logger.debug("Selector preferred call raised, trying fallbacks: %s", e)

        try:
            return list(selector.select(local_indices, labels=labels, num_poison=int(num_poison)))
        except TypeError:
            pass
        except Exception as e:
            logger.debug("Selector fallback (indices, labels, num_poison) raised: %s", e)

        try:
            return list(selector.select(local_indices, int(num_poison)))
        except Exception as e:
            logger.debug("Selector fallback (indices, num_poison) raised: %s", e)

        try:
            return list(selector.select(local_indices))
        except Exception as e:
            logger.debug("Selector fallback (indices) raised: %s", e)

        # final fallback: random sampling
        rng = np.random.RandomState(int(self.get_id()) + 0xDEADBEEF)
        num = min(int(num_poison), len(local_indices))
        return list(rng.choice(local_indices, size=num, replace=False))

    # ---------- normalize poisoner output ----------
    @staticmethod
    def _normalize_update(result: Any, client_id: int, num_samples: int) -> Dict[str, Any]:
        """
        Poisoner may return either:
         - a dict with keys 'weights' (state_dict), 'num_samples', 'client_id', 'metrics', ...
         - or a raw state_dict (then we wrap it).
        Ensure returned dict uses CPU tensors for 'weights'.
        """
        if isinstance(result, dict):
            upd = result
        elif isinstance(result, (tuple, list)) and len(result) >= 1 and isinstance(result[0], dict):
            upd = result[0]
        else:
            # assume it's a state_dict
            upd = {'weights': result, 'num_samples': num_samples, 'client_id': client_id, 'metrics': {}}

        # ensure weights are CPU clones
        if 'weights' in upd and isinstance(upd['weights'], dict):
            upd['weights'] = {k: v.cpu().clone() for k, v in upd['weights'].items()}
        return upd

    # ---------- selection helper ----------
    def _select_poison_indices(self, num_poison: Optional[int] = None) -> List[int]:
        """
        Return a list of indices (relative to local dataset) chosen to be poisoned.
        """
        if self.trainloader is None:
            return []

        local_dataset = self.trainloader.dataset
        N = len(local_dataset)
        if N == 0:
            return []

        local_indices = list(range(N))
        labels = None
        try:
            labels = self._get_labels_array(local_dataset)
        except Exception as e:
            logger.debug("Failed to extract labels from dataset: %s", e)
            labels = None

        # determine num_poison if not provided
        if num_poison is None:
            if isinstance(self.poison_fraction, float) and self.poison_fraction <= 1.0:
                num_poison = max(1, int(round(self.poison_fraction * N)))
            else:
                num_poison = int(self.poison_fraction)

        # selector absent -> random choice
        if self.selector is None:
            rng = np.random.RandomState(int(self.get_id()) + 0xC0FFEE)
            chosen = rng.choice(local_indices, size=min(num_poison, N), replace=False).tolist()
            return chosen

        # call selector (flexible)
        chosen = self._call_selector(self.selector, local_dataset, local_indices, num_poison, labels)
        # sanitize and cap
        chosen = [int(x) for x in chosen if 0 <= int(x) < N]
        if len(chosen) > num_poison:
            chosen = chosen[:num_poison]
        return chosen

    # ---------- main local_train ----------
    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Workflow:
          1. determine selected indices via selector (relative to local dataset)
          2. if poisoner provided: call poisoner.poison_and_train(client, selected_indices, trigger, global_params, epochs, round_idx)
             else: do naive poisoning (wrap dataset with PoisonedWrapper and call BenignClient.local_train)
          3. normalize and return an update dict
        """
        # 1) compute selected indices
        selected = self._select_poison_indices()

        # 2) if poisoner exists and exposes poison_and_train -> delegate
        if self.poisoner is not None and hasattr(self.poisoner, "poison_and_train"):
            try:
                global_params = self.get_params()
                result = self.poisoner.poison_and_train(
                    client=self,
                    selected_indices=selected,
                    trigger=self.trigger,
                    global_params=global_params,
                    epochs=epochs,
                    round_idx=round_idx
                )
                upd = self._normalize_update(result, client_id=self.get_id(), num_samples=self.num_samples())
                return upd
            except Exception as e:
                logger.exception("Poisoner.poison_and_train raised; falling back to naive poisoning. Error: %s", e)
                # continue to naive path

        # 3) naive poisoning path: wrap dataset and call BenignClient.local_train
        if self.trainloader is None:
            return super().local_train(epochs, round_idx)

        forced_label = int(self.target_label) if self.target_label is not None else None
        base_dataset = self.trainloader.dataset

        # selected is list of relative indices; PoisonedWrapper expects indices relative to base_dataset
        poisoned_rel = set(selected)
        poisoned_ds = PoisonedWrapper(base_dataset, poisoned_rel, trigger=self.trigger, forced_label=forced_label, transform_after=None)

        batch_size = getattr(self.trainloader, "batch_size", 32)
        shuffle = True
        num_workers = getattr(self.trainloader, "num_workers", 0)
        poisoned_loader = DataLoader(poisoned_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        orig_loader = self.trainloader
        try:
            self.trainloader = poisoned_loader
            result = super().local_train(epochs, round_idx)
        finally:
            self.trainloader = orig_loader

        # ensure weights are CPU clones
        if isinstance(result, dict) and 'weights' in result:
            result['weights'] = {k: v.cpu().clone() for k, v in result['weights'].items()}
        return result
