# src/attacks/components.py
from abc import ABC, abstractmethod
from typing import Sequence, List, Optional, Set, Dict, Any, Iterable, Union, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import copy as cp
import math

try:
    from PIL import Image
    import torchvision.transforms.functional as TF
except Exception:
    Image = None
    TF = None

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

# DistributedPatchTrigger
# Splits a single "full patch" into disjoint shards and applies one shard per sample.

class DistributedPatchTrigger:
    """
    A distributed trigger which defines one large "full patch" and splits it into shards.
    Each sample receives ONE shard (determined deterministically by sample_idx).
    - full_patch_size: int size (square) of the full patch in pixels (P).
    - shard_grid: (rows, cols) how to split the full patch into shards -> k = rows*cols
    - value: scalar or per-channel iterable used to fill the full patch if no explicit pattern provided.
    - pattern: optional pattern as PIL.Image / numpy array / torch.Tensor to use as full patch (will be resized to full_patch_size).
    - position: top-left (r, c) where the full patch would be anchored inside the image. If None, defaults to bottom-right.
    - behavior:
        - apply(x, sample_idx) -> inserts the shard corresponding to sample_idx into the image
        - apply(x) -> inserts the entire full patch (useful for previewing what the full patch looks like)
    """
    def __init__(
        self,
        full_patch_size: int = 12,
        shard_grid: Tuple[int,int] = (2, 2),
        value: Union[float,int, Iterable[float]] = 1.0,
        pattern: Optional[Union['Image.Image', np.ndarray, torch.Tensor]] = None,
        position: Optional[Tuple[int,int]] = None,
    ):
        self.P = int(full_patch_size)
        self.rows, self.cols = int(shard_grid[0]), int(shard_grid[1])
        assert self.rows > 0 and self.cols > 0, "shard_grid must be positive ints"
        self.k = self.rows * self.cols
        self.value = value
        self.pattern = pattern
        self.position = position  # top-left anchor for full patch

    def _make_full_patch_tensor(self, C: int, device=None, dtype=torch.float32) -> torch.Tensor:
        """
        Return full_patch tensor of shape (C, P, P) in [0,1]-like floats.
        If pattern provided, convert/resize; otherwise fill with `value`.
        """
        if self.pattern is not None:
            # handle various pattern types
            if isinstance(self.pattern, torch.Tensor):
                pt = self.pattern.clone().float()
                # expect (C,H,W) or (H,W) shape
                if pt.dim() == 3 and pt.shape[0] in (1,3):
                    # resize if necessary
                    if pt.shape[1:] != (self.P, self.P):
                        pt = torch.nn.functional.interpolate(pt.unsqueeze(0), size=(self.P, self.P), mode='bilinear', align_corners=False).squeeze(0)
                else:
                    # try (H,W) -> add channel
                    if pt.ndim == 2:
                        pt = pt.unsqueeze(0)
                        pt = torch.nn.functional.interpolate(pt.unsqueeze(0), size=(self.P, self.P), mode='bilinear', align_corners=False).squeeze(0)
                if pt.max() > 1.1:
                    pt = pt / 255.0
                if pt.shape[0] != C:
                    if pt.shape[0] == 1:
                        pt = pt.expand(C, -1, -1)
                    else:
                        pt = pt[:C, :, :]
                return pt.to(device=device, dtype=dtype)
            if Image is not None and isinstance(self.pattern, Image.Image):
                pil = self.pattern.resize((self.P, self.P), resample=Image.BILINEAR)
                arr = np.asarray(pil).astype(np.float32)
                if arr.max() > 1.1:
                    arr = arr / 255.0
                arr = arr.transpose(2, 0, 1)
                t = torch.from_numpy(arr).float()
                if t.shape[0] != C:
                    if t.shape[0] == 1:
                        t = t.expand(C, -1, -1)
                    else:
                        t = t[:C, :, :]
                return t.to(device=device, dtype=dtype)
            if isinstance(self.pattern, np.ndarray):
                arr = self.pattern.astype(np.float32)
                if arr.max() > 1.1:
                    arr = arr / 255.0
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, 2)
                arr = np.asarray(Image.fromarray((arr*255).astype(np.uint8)).resize((self.P, self.P))).astype(np.float32) / 255.0
                arr = arr.transpose(2, 0, 1)
                t = torch.from_numpy(arr).float()
                if t.shape[0] != C:
                    if t.shape[0] == 1:
                        t = t.expand(C, -1, -1)
                    else:
                        t = t[:C, :, :]
                return t.to(device=device, dtype=dtype)

        # no pattern: fill with scalar/per-channel value
        if isinstance(self.value, (int, float)):
            fill = float(self.value)
            t = torch.full((C, self.P, self.P), fill, dtype=dtype, device=device)
            return t
        else:
            vals = list(self.value)
            if len(vals) < C:
                vals = vals + [vals[-1]] * (C - len(vals))
            t = torch.zeros((C, self.P, self.P), dtype=dtype, device=device)
            for ch in range(C):
                t[ch] = float(vals[ch])
            return t

    def _split_to_shards(self, full_patch: torch.Tensor) -> list:
        """Split (C,P,P) into shards list of (C, shard_h, shard_w) in row-major order."""
        C, P, _ = full_patch.shape
        shard_h = P // self.rows
        shard_w = P // self.cols
        shards = []
        for r in range(self.rows):
            for c in range(self.cols):
                r0 = r * shard_h
                c0 = c * shard_w
                # for boundary shards include remainder pixels
                r1 = (r + 1) * shard_h if r < self.rows - 1 else P
                c1 = (c + 1) * shard_w if c < self.cols - 1 else P
                shard = full_patch[:, r0:r1, c0:c1].clone()
                shards.append(shard)
        return shards

    def _place_shard_tensor(self, img: torch.Tensor, shard: torch.Tensor, base_r: int, base_c: int) -> torch.Tensor:
        """
        Place shard (C, sh, sw) into image tensor img (C,H,W) at offset (base_r, base_c).
        This will paste shard into img at [base_r:base_r+sh, base_c:base_c+sw], clamped by image bounds.
        """
        xt = img.clone()
        C, H, W = xt.shape
        sh_ch, sh_h, sh_w = shard.shape
        assert sh_ch == C or sh_ch == 1
        dst_r0 = max(0, base_r)
        dst_c0 = max(0, base_c)
        dst_r1 = min(H, base_r + sh_h)
        dst_c1 = min(W, base_c + sh_w)
        src_r0 = dst_r0 - base_r
        src_c0 = dst_c0 - base_c
        src_r1 = src_r0 + (dst_r1 - dst_r0)
        src_c1 = src_c0 + (dst_c1 - dst_c0)
        piece = shard[:, src_r0:src_r1, src_c0:src_c1]
        if piece.shape[0] != C:
            if piece.shape[0] == 1:
                piece = piece.expand(C, -1, -1)
            else:
                piece = piece[:C, :, :]
        xt[:, dst_r0:dst_r1, dst_c0:dst_c1] = piece
        return xt

    def _apply_to_tensor(self, x: torch.Tensor, sample_idx: Optional[int] = None) -> torch.Tensor:
        # x: (C, H, W), values arbitrary range; we treat patch as same scale e.g., [0,1] for normalized tensors
        C, H, W = x.shape
        device = x.device
        # build full patch tensor
        full_patch = self._make_full_patch_tensor(C, device=device, dtype=torch.float32)
        shards = self._split_to_shards(full_patch)
        # determine anchor position for full patch top-left
        if self.position is None:
            base_r = max(0, H - self.P - 1)
            base_c = max(0, W - self.P - 1)
        else:
            base_r, base_c = int(self.position[0]), int(self.position[1])

        if sample_idx is None:
            # preview: paste full patch by layering shards
            xt = x.clone()
            # place shards in their correct offset
            shard_h_approx = self.P // self.rows
            shard_w_approx = self.P // self.cols
            idx = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    r_off = base_r + r * shard_h_approx
                    c_off = base_c + c * shard_w_approx
                    xt = self._place_shard_tensor(xt, shards[idx], r_off, c_off)
                    idx += 1
            return xt
        # choose which shard to apply based on sample_idx deterministically
        sid = int(sample_idx) % self.k
        shard = shards[sid]
        # compute shard's offset within full patch:
        shard_h_approx = self.P // self.rows
        shard_w_approx = self.P // self.cols
        r_idx = sid // self.cols
        c_idx = sid % self.cols
        base_r_sh = base_r + r_idx * shard_h_approx
        base_c_sh = base_c + c_idx * shard_w_approx
        return self._place_shard_tensor(x, shard, base_r_sh, base_c_sh)

    def _apply_to_pil(self, img: 'Image.Image', sample_idx: Optional[int] = None):
        if Image is None:
            raise RuntimeError("PIL not available")
        arr = np.asarray(img.convert("RGB")).astype(np.float32)
        H, W = arr.shape[:2]
        C = 3
        # convert to tensor in [0,1]
        t = torch.from_numpy(arr.transpose(2,0,1)).float() / 255.0
        out = self._apply_to_tensor(t, sample_idx)
        out_np = (out.cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8)
        return Image.fromarray(out_np)

    def apply(self, x, sample_idx: Optional[int] = None):
        """
        Apply distributed fragment patch. If sample_idx is provided, apply only the shard for that sample.
        If sample_idx is None, paste the full patch (all shards) for preview.
        """
        # accept both tensor and PIL
        if isinstance(x, torch.Tensor):
            # if input is integer dtype, convert to float in [0,1], then map back? We preserve dtype by working in float.
            orig_dtype = x.dtype
            xt = x.float()
            res = self._apply_to_tensor(xt, sample_idx)
            # try to cast back if original was integer
            if not torch.is_floating_point(x):
                # assume 0..255 original
                res = (res * 255.0).clamp(0, 255).to(orig_dtype)
            return res
        if Image is not None and isinstance(x, Image.Image):
            return self._apply_to_pil(x, sample_idx)
        # fallback: try to convert via torchvision
        if TF is not None:
            xt = TF.to_tensor(x)
            rt = self._apply_to_tensor(xt, sample_idx)
            return TF.to_pil_image(rt)
        raise TypeError("Unsupported input type for DistributedFragmentPatchTrigger.apply")



class BlendedTrigger:
    """
    Blend an additive pattern image into the whole image (global blended trigger).

    Parameters:
    - pattern: optional pattern provided as PIL.Image or torch.Tensor (C,H,W) or numpy array (H,W,C).
               If None, a default checkerboard/noise pattern is created.
    - alpha: blend strength in [0,1]. Result = (1-alpha) * x + alpha * pattern
    - normalize_input: if True and input is float tensor normalized outside [0,1], user must ensure alpha/value consistent
    - resize_pattern: if True, pattern will be resized to input HxW as needed
    """
    def __init__(self, pattern= None, alpha: float = 0.15, resize_pattern: bool = True, seed: int = 0):
        self.alpha = float(alpha)
        self.resize_pattern = bool(resize_pattern)
        self.seed = int(seed)
        self.pattern = pattern
        if self.pattern is None:
            self.pattern = self._make_checkerboard()

    def _make_checkerboard(self, size: int = 32, tile: int = 4):
        # default pattern small; will be resized to image by apply
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        t = max(1, tile)
        for i in range(size):
            for j in range(size):
                if ((i // t) + (j // t)) % 2 == 0:
                    arr[i, j] = [255, 255, 255]
                else:
                    arr[i, j] = [0, 0, 0]
        if Image is not None:
            return Image.fromarray(arr)
        return arr

    def _to_tensor_pattern(self, H: int, W: int, device=None, dtype=torch.float32):
        # return pattern as torch.Tensor (C,H,W) float in [0,1]
        p = self.pattern
        if isinstance(p, torch.Tensor):
            pt = p.clone().float()
            if pt.dim() == 3 and pt.shape[0] in (1,3):
                # (C,H,W)
                if pt.shape[1:] != (H, W) and self.resize_pattern:
                    pt = torch.nn.functional.interpolate(pt.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
                # ensure in [0,1] if dtype suggests so
                if pt.max() > 1.1:
                    pt = pt / 255.0
                return pt.to(device=device, dtype=dtype)
            elif pt.ndim == 2:
                # (H,W)
                pt = pt.unsqueeze(0)
                if self.resize_pattern and pt.shape[1:] != (H, W):
                    pt = torch.nn.functional.interpolate(pt.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
                if pt.max() > 1.1:
                    pt = pt / 255.0
                return pt.to(device=device, dtype=dtype)
        if Image is not None and isinstance(p, Image.Image):
            pil = p
            if self.resize_pattern and pil.size != (W, H):
                pil = pil.resize((W, H), resample=Image.BILINEAR)
            arr = np.asarray(pil).astype(np.float32)
            if arr.max() > 1.1:
                arr = arr / 255.0
            arr = arr.transpose(2, 0, 1)  # C,H,W
            return torch.from_numpy(arr).float().to(device=device, dtype=dtype)
        # numpy array
        if isinstance(p, np.ndarray):
            arr = p.astype(np.float32)
            if arr.max() > 1.1:
                arr = arr / 255.0
            if arr.ndim == 2:
                arr = np.expand_dims(arr, 2)
            arr = arr.transpose(2, 0, 1)
            # resize if needed
            if self.resize_pattern and (arr.shape[1] != H or arr.shape[2] != W):
                pt = torch.from_numpy(arr).float().unsqueeze(0)
                pt = torch.nn.functional.interpolate(pt, size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
                return pt
            return torch.from_numpy(arr).float()
        raise RuntimeError("Unsupported pattern type for BlendedTrigger")

    def _apply_tensor(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W), float or other dtype. We try to keep dtype but compute in float.
        C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        x_float = x.float()
        pat = self._to_tensor_pattern(H, W, device=device, dtype=torch.float32)
        # if pattern has single channel expand to input channels
        if pat.shape[0] == 1 and C > 1:
            pat = pat.expand(C, -1, -1)
        if pat.shape[0] != C:
            # if mismatch try to broadcast/truncate/expand
            if pat.shape[0] > C:
                pat = pat[:C, :, :]
            else:
                pat = pat.expand(C, -1, -1)
        out = (1.0 - self.alpha) * x_float + self.alpha * pat.to(device=device)
        # cast back to original dtype range if needed
        if dtype.is_floating_point:
            return out.to(dtype)
        else:
            # integer dtype: assume original was 0..255
            out = (out * 255.0).clamp(0, 255).to(dtype)
            return out

    def _apply_pil(self, img):
        if Image is None:
            raise RuntimeError("PIL not available")
        H, W = img.size[1], img.size[0]
        pat = self.pattern
        pil_pat = pat
        if isinstance(pat, Image.Image):
            if self.resize_pattern and (pil_pat.size != (W, H)):
                pil_pat = pil_pat.resize((W, H), resample=Image.BILINEAR)
            arr_img = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
            arr_pat = np.asarray(pil_pat.convert("RGB")).astype(np.float32) / 255.0
            blended = (1.0 - self.alpha) * arr_img + self.alpha * arr_pat
            blended = (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)
            return Image.fromarray(blended)
        else:
            # fallback: convert to tensor and use tensor path
            if TF is None:
                raise RuntimeError("tvf not available to convert images")
            xt = TF.to_tensor(img)
            xt = self._apply_tensor(xt)
            return TF.to_pil_image(xt)

    def apply(self, x):
        """
        Apply blended trigger. Accepts torch.Tensor(C,H,W) or PIL.Image.
        """
        if isinstance(x, torch.Tensor):
            return self._apply_tensor(x)
        if Image is not None and isinstance(x, Image.Image):
            return self._apply_pil(x)
        # fallback to tensor conversion if possible
        if TF is not None:
            xt = TF.to_tensor(x)
            xt = self._apply_tensor(xt)
            return TF.to_pil_image(xt)
        raise TypeError("Unsupported input type for BlendedTrigger.apply")

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
