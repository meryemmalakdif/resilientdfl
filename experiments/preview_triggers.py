#!/usr/bin/env python3
"""
scripts/preview_fashionmnist_triggers.py

Preview triggers applied to FashionMNIST samples.

Examples:
  # 16 images with a 4x4 patch trigger at bottom-right
  python scripts/preview_fashionmnist_triggers.py --trigger patch --patch-size 4 --num 16 --out out/patch_preview.png

  # distributed shard trigger: 8x8 full patch split into 2x2 shards; preview for client 2
  python scripts/preview_fashionmnist_triggers.py --trigger distributed_shard --full-size 8 --shard-rows 2 --shard-cols 2 --trigger-loc 10,10 --client-id 2 --num 9 --out out/dist_preview.png

  # save masks as a second image
  python scripts/preview_fashionmnist_triggers.py --trigger patch --patch-size 4 --num 9 --out out/patch.png --save-masks
"""
import argparse
import os
import random
from typing import Optional, Sequence, Tuple, List, Union
import math

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

# Try to import user triggers from common locations
PatchTrigger = None
DistributedShardTrigger = None
_import_errors = []
for mod in ("trigger", "src.attacks.triggers", "src.attacks.trigger"):
    try:
        m = __import__(mod, fromlist=["PatchTrigger", "DistributedShardTrigger", "DistributedTrigger"])
        if PatchTrigger is None:
            PatchTrigger = getattr(m, "PatchTrigger", PatchTrigger)
        if DistributedShardTrigger is None:
            DistributedShardTrigger = getattr(m, "DistributedShardTrigger", getattr(m, "DistributedTrigger", None))
        if PatchTrigger is not None and DistributedShardTrigger is not None:
            break
    except Exception as e:
        _import_errors.append((mod, e))

if PatchTrigger is None:
    raise ImportError("Could not import PatchTrigger. Make sure trigger.py (or src.attacks.triggers) is in PYTHONPATH and defines PatchTrigger.")

if DistributedShardTrigger is None:
    raise ImportError("Could not import DistributedShardTrigger/DistributedTrigger. Make sure trigger.py (or src.attacks.triggers) is in PYTHONPATH and defines DistributedShardTrigger or DistributedTrigger.")


def parse_int_pair(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if s is None:
        return None
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) == 1:
        return int(parts[0]), int(parts[0])
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("Expected one or two integers separated by comma")


def build_trigger(args, sample_img: Optional[torch.Tensor] = None):
    """
    Build and return a trigger instance compatible with your trigger API.
    sample_img (C,H,W) may be used to size learnable patches when needed.
    """
    ttype = args.trigger.lower()
    if ttype == "patch":
        # PatchTrigger(patch_size=..., position=..., value=..., alpha=...)
        position = args.position
        return PatchTrigger(
            patch_size=args.patch_size,
            position=position,
            value=args.value,
            alpha=args.alpha
        )

    if ttype == "distributed_shard" or ttype == "distributed":
        # Use DistributedShardTrigger signature expected in earlier message:
        # DistributedShardTrigger(malicious_clients=[...], client_id=..., full_size=(h,w), shard_grid=(rows,cols), trigger_loc=(top,left), value=..., alpha=...)
        full_size = (args.full_size, args.full_size) if args.full_size_single else (args.full_h, args.full_w)
        shard_grid = (args.shard_rows, args.shard_cols)
        # prepare malicious clients list if provided, else None (mapping will use client_id % num_shards)
        malicious = None
        if args.malicious_clients:
            # parse comma-separated list of ints
            malicious = [int(x.strip()) for x in args.malicious_clients.split(",") if x.strip() != ""]
        # trigger_loc parse
        loc = (0, 0)
        if args.trigger_loc:
            loc = tuple(map(int, args.trigger_loc.split(",")))
        return DistributedShardTrigger(
            malicious_clients=malicious,
            client_id=args.client_id if args.client_id is not None else -1,
            full_size=full_size,
            shard_grid=shard_grid,
            trigger_loc=loc,
            value=args.value,
            alpha=args.alpha,
        )

    raise ValueError(f"Unknown trigger type: {args.trigger}")


def undo_normalize_tensor(x: torch.Tensor, mean: Optional[Sequence[float]], std: Optional[Sequence[float]]) -> torch.Tensor:
    xt = x.clone()
    if mean is None or std is None:
        return xt
    mean_t = torch.tensor(mean, device=xt.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=xt.device).view(-1, 1, 1)
    return xt * std_t + mean_t


def ensure_01_tensor(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(x):
        x = x.float() / 255.0
    return x.clamp(0.0, 1.0)


def sample_indices(dataset_len: int, n: int, seed: int):
    rng = np.random.RandomState(seed)
    n = min(n, dataset_len)
    return rng.choice(np.arange(dataset_len), size=n, replace=False).tolist()


def make_mask_grid(masks: List[torch.Tensor], nrow: int = 0):
    """
    Build a visual grid for single-channel masks: convert to 3-channel red overlay visualization.
    masks: list of (1,H,W) tensors in [0,1]
    returns a (3,H_grid,W_grid) tensor in [0,1]
    """
    rgb_masks = []
    for m in masks:
        if m.dim() == 3:
            m = m.squeeze(0)
        # make red overlay: R = m, G = 0, B = 0
        r = m
        g = torch.zeros_like(m)
        b = torch.zeros_like(m)
        rgb = torch.stack([r, g, b], dim=0)  # (3,H,W)
        rgb_masks.append(rgb)
    if len(rgb_masks) == 0:
        return None
    if nrow <= 0:
        nrow = int(math.ceil(math.sqrt(len(rgb_masks))))
    grid = make_grid(rgb_masks, nrow=nrow, pad_value=1.0)
    return grid


def preview(args):
    # prepare dataset (FashionMNIST)
    transform = transforms.ToTensor()  # returns (C,H,W) in [0,1], C=1
    ds = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform)
    dataset_len = len(ds)

    indices = sample_indices(dataset_len, args.num, args.seed)
    # build trigger instance; if trigger needs sample shape, pass sample later
    # get a sample to determine channels/shape (we'll pass to builder only if necessary)
    sample_img, _ = ds[indices[0]]
    trigger = build_trigger(args, sample_img)

    poisoned_images = []
    masks = []

    for idx in indices:
        x, y = ds[idx]  # x is (1,H,W) float in [0,1]
        if not torch.is_floating_point(x):
            x = x.float() / 255.0
        # ensure shape (C,H,W)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # apply trigger (expect trigger.apply(x) -> (poisoned, mask) OR trigger.apply accepts batch)
        try:
            poisoned, mask = trigger.apply(x)  # prefer tensor API
        except TypeError:
            # some triggers may use 'apply' that accepts batch or returns images directly
            out = trigger.apply(x)
            if isinstance(out, tuple) and len(out) == 2:
                poisoned, mask = out
            else:
                poisoned = out
                # create dummy mask of zeros
                mask = torch.zeros((1, x.shape[1], x.shape[2]), dtype=x.dtype)

        # undo normalization if requested
        mean = None
        std = None
        if args.undo_mean and args.undo_std:
            mean = [float(m) for m in args.undo_mean.split(",")]
            std = [float(s) for s in args.undo_std.split(",")]
            poisoned = undo_normalize_tensor(poisoned, mean, std)

        poisoned = ensure_01_tensor(poisoned)
        # ensure mask is (1,H,W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        masks.append(mask)
        poisoned_images.append(poisoned)

    # make grids and save
    n = len(poisoned_images)
    nrow = int(math.ceil(math.sqrt(n)))
    grid_img = make_grid(poisoned_images, nrow=nrow, pad_value=1.0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(grid_img, args.out)
    print(f"Saved triggered image grid ({n} samples) to: {args.out}")

    if args.save_masks:
        mask_out = args.out.replace(".png", "_masks.png")
        mask_grid = make_mask_grid(masks, nrow=nrow)
        if mask_grid is not None:
            save_image(mask_grid, mask_out)
            print(f"Saved mask grid to: {mask_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data", help="where to download FashionMNIST")
    p.add_argument("--num", type=int, default=9, help="number of images to preview")
    p.add_argument("--seed", type=int, default=0, help="seed for sampling images")
    p.add_argument("--trigger", type=str, default="patch", choices=["patch", "distributed_shard", "distributed"], help="which trigger to preview")
    p.add_argument("--patch-size", type=int, default=4, help="patch size (used by patch trigger)")
    p.add_argument("--position", type=str, default="bottom_right", help="position spec for patch trigger (use keywords: bottom_right, top_left, center, random)")
    p.add_argument("--value", type=float, default=1.0, help="patch replacement value in [0,1] (or scalar normalized value). For grayscale FashionMNIST, a scalar is fine.")
    p.add_argument("--alpha", type=float, default=1.0, help="blend alpha in [0,1]")
    # distributed shard options
    p.add_argument("--full-size", type=int, default=None, help="full contiguous patch size (square) to shard (overrides full-h/full-w if set)")
    p.add_argument("--full-h", type=int, default=8, help="full patch height (used if --full-size not given)")
    p.add_argument("--full-w", type=int, default=8, help="full patch width (used if --full-size not given)")
    p.add_argument("--full-size-single", dest="full_size_single", action="store_true", help="interpret --full-size as single int for both dims", default=True)
    p.add_argument("--shard-rows", type=int, default=2, help="shard grid rows")
    p.add_argument("--shard-cols", type=int, default=2, help="shard grid cols")
    p.add_argument("--trigger-loc", type=str, default="20,20", help="top-left location for the full patch as 'row,col' (pixels)")
    p.add_argument("--malicious-clients", type=str, default=None, help="comma-separated malicious client ids (optional)")
    p.add_argument("--client-id", type=int, default=None, help="client id to preview (if omitted, preview server aggregated view -1)")
    p.add_argument("--out", type=str, default="data/fashionmnist_trigger_preview.png", help="where to save triggered grid")
    p.add_argument("--save-masks", action="store_true", help="also save masks grid (red overlay)")
    p.add_argument("--undo-mean", type=str, default=None, help="comma-separated mean to undo normalization for visualization (e.g. 0.5)")
    p.add_argument("--undo-std", type=str, default=None, help="comma-separated std to undo normalization for visualization (e.g. 0.5)")
    args = p.parse_args()

    # convenience mapping / normalize argument compatibility
    # default: if full-size provided, use it for both dims
    if args.full_size is not None:
        args.full_size_single = True
        args.full_h = args.full_size
        args.full_w = args.full_size
    # pick client id default: -1(server) if not provided
    if args.client_id is None:
        args.client_id = -1

    # map simple patch args into PatchTrigger expected parameters
    args.patch_size = args.patch_size
    args.position = args.position
    args.value = args.value
    args.alpha = args.alpha

    # call preview
    preview(args)


if __name__ == "__main__":
    main()
