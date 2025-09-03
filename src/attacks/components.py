# src/attacks/components.py
from abc import ABC, abstractmethod
from typing import Sequence, List, Optional, Set, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import copy as cp

# import BenignClient to reuse its training method (avoid recursion)
from src.fl.baseclient import BenignClient


# ----------------------------
# SELECTOR INTERFACE + EXAMPLE
# ----------------------------
class Selector(ABC):
    """Select indices (relative to a local dataset) to poison."""
    @abstractmethod
    def select(self, dataset: Dataset, indices: Sequence[int], num_poison: int, labels: Optional[Sequence[int]] = None) -> List[int]:
        """
        Return a list of indices (subset of `indices`) to poison.
        - dataset: dataset object (can be used for feature extraction)
        - indices: list of local indices (0..N-1 or subset mapping)
        - num_poison: desired number of poisoned samples (absolute). Implementations may treat floats specially.
        - labels: optional array-like of labels aligned with dataset (useful for label-aware selectors)
        """
        raise NotImplementedError()


class RandomSelector(Selector):
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    def select(self, dataset: Dataset, indices: Sequence[int], num_poison: int, labels: Optional[Sequence[int]] = None) -> List[int]:
        idxs = list(indices)
        if num_poison >= len(idxs):
            return idxs
        chosen = self.rng.choice(idxs, size=num_poison, replace=False).tolist()
        return [int(x) for x in chosen]





# ----------------------------
# TRIGGER INTERFACE + PATCH TRIGGER
# ----------------------------
class Trigger(ABC):
    """Applies a trigger to a single example (PIL or Tensor) and returns the same type."""
    @abstractmethod
    def apply(self, x):
        """Return triggered x (should match type/shape of input)."""
        raise NotImplementedError()


class PatchTrigger(Trigger):
    """
    Simple square patch trigger that supports both PIL.Image and torch.Tensor inputs.
    - patch_size in pixels (int)
    - value: for tensor inputs in [0,1] use 1.0, for uint8 use 255; can be scalar or tuple/channel-wise
    - position: None -> bottom-right (default), else (row, col)
    """
    def __init__(self, patch_size: int = 3, value=1.0, position: Optional[tuple] = None):
        self.patch_size = int(patch_size)
        self.value = value
        self.position = position

    def apply(self, x):
        # handle tensor input (C,H,W)
        if isinstance(x, torch.Tensor):
            xt = x.clone()
            C, H, W = xt.shape
            ps = self.patch_size
            if self.position is None:
                r = H - ps - 1
                c = W - ps - 1
            else:
                r, c = self.position
            # if value is scalar or iterable
            if isinstance(self.value, (int, float)):
                fill = float(self.value)
                xt[:, r:r+ps, c:c+ps] = fill
            else:
                # assume per-channel iterable
                for ch in range(C):
                    xt[ch, r:r+ps, c:c+ps] = float(self.value[ch])
            return xt

        # fall back to PIL (lazy import)
        try:
            from PIL import Image
            import numpy as np
        except Exception:
            raise RuntimeError("PIL not available for PatchTrigger on PIL inputs.")
        if isinstance(x, Image.Image):
            arr = np.array(x.convert("RGB"))
            H, W = arr.shape[:2]
            ps = self.patch_size
            if self.position is None:
                r = H - ps - 1
                c = W - ps - 1
            else:
                r, c = self.position
            arr[r:r+ps, c:c+ps] = self.value
            return Image.fromarray(arr)
        raise TypeError("Unsupported input type for PatchTrigger.apply")


# ----------------------------
# POISONER INTERFACE + IMPLEMENTATIONS
# ----------------------------
class Poisoner(ABC):
    """
    Model-poisoning strategy.
    poison_and_train(...) must return the standard client update dict:
      {'client_id', 'num_samples', 'weights', 'metrics', 'round_idx'}
    """
    @abstractmethod
    def poison_and_train(self, client: BenignClient, selected_indices: Sequence[int], trigger: Trigger,
                         global_params: Dict[str, torch.Tensor], epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()


# helper: create poisoned wrapper dataset (non-destructive)
class PoisonedWrapper(Dataset):
    def __init__(self, base_dataset: Dataset, poisoned_rel_indices: Set[int], trigger: Trigger, forced_label: Optional[int] = None, transform_after=None):
        self.base = base_dataset
        self.poisoned = set(poisoned_rel_indices)
        self.trigger = trigger
        self.forced_label = forced_label
        self.transform_after = transform_after if transform_after is not None else getattr(base_dataset, "transform", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.poisoned:
            x = self.trigger.apply(x)
            if self.forced_label is not None:
                y = self.forced_label
        # if base didn't apply transforms but transform_after is set, apply it
        if getattr(self.base, "transform", None) is None and self.transform_after is not None:
            return self.transform_after(x), y
        return x, y


def _state_dict_to_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.cpu().clone() for k, v in sd.items()}


def _dict_delta(local: Dict[str, torch.Tensor], globalp: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (local[k].cpu().float() - globalp[k].cpu().float()) for k in local.keys()}


def _add_scaled_delta(globalp: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    return {k: (globalp[k].cpu().float() + scale * delta[k]) for k in delta.keys()}


def _l2_norm_of_delta(delta: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for v in delta.values():
        total += float((v.view(-1) ** 2).sum().item())
    return float(np.sqrt(total))


def _clip_delta(delta: Dict[str, torch.Tensor], max_norm: float) -> Dict[str, torch.Tensor]:
    norm = _l2_norm_of_delta(delta)
    if norm <= max_norm:
        return delta
    scale = max_norm / (norm + 1e-12)
    return {k: v * scale for k, v in delta.items()}


class NaivePoisoner(Poisoner):
    """
    Naive data-poisoning: train on poisoned dataset (poisoned + clean samples) with client's usual training logic.
    Implementation details:
      - It loads global params into client's model (client.set_params should exist)
      - Builds poisoned DataLoader by wrapping client's trainloader.dataset with PoisonedWrapper
      - Calls BenignClient.local_train(client, epochs, round_idx) directly (bypasses MaliciousClient override)
      - Returns the update dict returned by BenignClient.local_train (weights are CPU tensors)
    """
    def __init__(self, forced_label: Optional[int] = None, reapply_transform_after=None):
        self.forced_label = forced_label
        self.transform_after = reapply_transform_after

    def poison_and_train(self, client: BenignClient, selected_indices: Sequence[int], trigger: Trigger,
                         global_params: Dict[str, torch.Tensor], epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        # ensure client's model initialized with global params
        client.set_params(global_params)

        base_dataset = client.trainloader.dataset
        poisoned_rel = set(int(i) for i in selected_indices)

        poisoned_ds = PoisonedWrapper(base_dataset, poisoned_rel, trigger, forced_label=self.forced_label, transform_after=self.transform_after)
        # keep same loader options
        batch_size = getattr(client.trainloader, "batch_size", 32)
        num_workers = getattr(client.trainloader, "num_workers", 0)
        poisoned_loader = DataLoader(poisoned_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # swap loaders safely and call the benign client's training logic
        orig_loader = client.trainloader
        try:
            client.trainloader = poisoned_loader
            # call BenignClient.local_train directly to avoid recursion (MaliciousClient will delegate here)
            result = BenignClient.local_train(client, epochs, round_idx)
        finally:
            client.trainloader = orig_loader

        # ensure weights are CPU clones
        result['weights'] = _state_dict_to_cpu(result['weights'])
        return result


class ModelReplacementPoisoner(Poisoner):
    """
    After doing local training on poisoned data (via NaivePoisoner), scale the delta so attacker can replace global model.
    new_weights = global + gamma * (local - global)
    """
    def __init__(self, gamma: float = 1.0, forced_label: Optional[int] = None):
        self.gamma = float(gamma)
        self.forced_label = forced_label
        self.naive = NaivePoisoner(forced_label=forced_label)

    def poison_and_train(self, client: BenignClient, selected_indices: Sequence[int], trigger: Trigger,
                         global_params: Dict[str, torch.Tensor], epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        # first obtain local trained update
        local_update = self.naive.poison_and_train(client, selected_indices, trigger, global_params, epochs, round_idx, **kwargs)
        local_weights = local_update['weights']
        # compute delta
        delta = _dict_delta(local_weights, global_params)
        # scale and add
        replaced = _add_scaled_delta(global_params, delta, self.gamma)
        local_update['weights'] = {k: v.clone() for k, v in replaced.items()}
        return local_update


class ConstrainedPoisoner(Poisoner):
    """
    Train normally on poisoned data then constrain the model delta to have L2 norm <= max_norm.
    Optionally scale after clipping by factor `scale_after_clip`.
    """
    def __init__(self, max_norm: float = 1.0, scale_after_clip: float = 1.0, forced_label: Optional[int] = None):
        self.max_norm = float(max_norm)
        self.scale_after_clip = float(scale_after_clip)
        self.forced_label = forced_label
        self.naive = NaivePoisoner(forced_label=forced_label)

    def poison_and_train(self, client: BenignClient, selected_indices: Sequence[int], trigger: Trigger,
                         global_params: Dict[str, torch.Tensor], epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        local_update = self.naive.poison_and_train(client, selected_indices, trigger, global_params, epochs, round_idx, **kwargs)
        local_weights = local_update['weights']
        delta = _dict_delta(local_weights, global_params)
        clipped = _clip_delta(delta, self.max_norm)
        scaled = _add_scaled_delta(global_params, clipped, self.scale_after_clip)
        local_update['weights'] = {k: v.clone() for k, v in scaled.items()}
        return local_update
