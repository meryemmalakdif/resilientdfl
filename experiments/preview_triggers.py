#!/usr/bin/env python3
"""
scripts/preview_trigger.py

Preview poisoned images for a chosen client and trigger.

Usage examples:

# Preview 16 images from client 0 with a 3x3 patch trigger:
PYTHONPATH=./src python scripts/preview_trigger.py --client 0 --num 16 --trigger patch --patch-size 3 --out preview_patch.png

# Preview with distributed random patches:
PYTHONPATH=./src python scripts/preview_trigger.py --client 2 --num 12 --trigger distributed --patch-size 2 --placement random --num-patches 6 --out preview_dist.png

# Preview blended trigger (checkerboard default pattern):
PYTHONPATH=./src python scripts/preview_trigger.py --client 1 --num 9 --trigger blended --alpha 0.12 --out preview_blend.png

# If your dataset loader returns normalized tensors, undo normalize for visualization:
PYTHONPATH=./src python scripts/preview_trigger.py --client 0 --num 9 --trigger patch --patch-size 3 --undo-mean 0.5,0.5,0.5 --undo-std 0.5,0.5,0.5

"""
import argparse
import os
import random
from typing import Optional, Sequence, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image

# try to import adapters and triggers from your project
try:
    from src.datasets.femnist import FEMNISTAdapter
except Exception:
    FEMNISTAdapter = None

# triggers: try unified components or separate module fallbacks
try:
    from src.attacks.components import PatchTrigger, DistributedPatchTrigger, BlendedTrigger
except Exception:
    raise ImportError("Could not import triggers from src.attacks.components")

def parse_floats(s: str) -> Optional[Sequence[float]]:
    if s is None or s.strip() == "":
        return None
    return [float(x.strip()) for x in s.split(",")]

def build_trigger(args):
    ttype = args.trigger.lower()
    if ttype == "patch":
        if PatchTrigger is None:
            raise RuntimeError("PatchTrigger not available in src.attacks.components or src.attacks.triggers")
        return PatchTrigger(patch_size=args.patch_size, value=args.value, position=None)
    if ttype == "distributed":
        if DistributedPatchTrigger is None:
            raise RuntimeError("DistributedPatchTrigger not available in src.attacks.components or src.attacks.triggers")
        return DistributedPatchTrigger(
            full_patch_size=args.patch_size,
            shard_grid=(3,3),  # fixed 3x3 grid for distributed trigger
            value=args.value,
        )
    if ttype == "blended":
        if BlendedTrigger is None:
            raise RuntimeError("BlendedTrigger not available in src.attacks.components or src.attacks.triggers")
        # if user provided a pattern path, load it
        if args.pattern is not None:
            pat = Image.open(args.pattern).convert("RGB")
        else:
            pat = None
        return BlendedTrigger(pattern=pat, alpha=args.alpha, resize_pattern=True, seed=args.seed)
    raise ValueError(f"Unknown trigger type: {args.trigger}")

def undo_normalize_tensor(x: torch.Tensor, mean: Optional[Sequence[float]], std: Optional[Sequence[float]]) -> torch.Tensor:
    """
    Undo normalization for a tensor image (C,H,W).
    mean/std are sequences either for single channel or per-channel.
    """
    xt = x.clone()
    if mean is None or std is None:
        return xt
    mean_t = torch.tensor(mean, device=xt.device).view(-1,1,1)
    std_t = torch.tensor(std, device=xt.device).view(-1,1,1)
    # xt is assumed normalized as (x - mean) / std -> recover x = xt * std + mean
    xt = xt * std_t + mean_t
    return xt

def ensure_01_tensor(x: torch.Tensor) -> torch.Tensor:
    """Convert tensor to float in [0,1] suitable for save_image"""
    if not torch.is_floating_point(x):
        x = x.float() / 255.0
    # clamp
    x = x.clamp(0.0, 1.0)
    return x

def load_client_loader(adapter, client_id: Optional[int], num_clients: int, batch_size: int):
    """
    Prefer adapter.get_client_loaders(num_clients,..). If not present, IID-split adapter.dataset.
    Returns dict client_id->dataloader.
    """
    if hasattr(adapter, "get_client_loaders"):
        try:
            loaders = adapter.get_client_loaders(num_clients=num_clients, strategy="iid", batch_size=batch_size)
            return loaders
        except TypeError:
            # try positional
            loaders = adapter.get_client_loaders(num_clients, "iid", batch_size)
            return loaders
    if hasattr(adapter, "dataset"):
        ds = adapter.dataset
        N = len(ds)
        idx = np.arange(N)
        np.random.shuffle(idx)
        splits = np.array_split(idx, num_clients)
        loaders = {}
        for i, s in enumerate(splits):
            loaders[i] = DataLoader(Subset(ds, list(map(int, s))), batch_size=batch_size, shuffle=True, num_workers=2)
        return loaders
    raise RuntimeError("Adapter must expose get_client_loaders or dataset")

def preview_trigger_for_client(args):
    # instantiate adapter
    if FEMNISTAdapter is None:
        raise RuntimeError("FEMNISTAdapter not found at src.datasets.femnist - adapt script to your dataset adapter.")
    try:
        train_adapter = FEMNISTAdapter(root="data", train=True, download=False)
    except TypeError:
        train_adapter = FEMNISTAdapter()

    num_clients = args.num_clients
    batch_size = max(1, args.batch)
    loaders = load_client_loader(train_adapter, args.client, num_clients=num_clients, batch_size=batch_size)

    # pick client id
    client_id = args.client if args.client is not None else random.choice(list(loaders.keys()))
    if client_id not in loaders:
        raise ValueError(f"client id {client_id} not present; available: {list(loaders.keys())}")
    loader = loaders[client_id]

    # collect samples from underlying dataset (prefer sampling directly from dataset to get per-index control)
    # If loader.dataset is a Subset, we want to sample indices relative to that Subset.
    ds = loader.dataset
    N = len(ds)
    n = min(args.num, N)
    # choose indices to preview (random)
    rng = np.random.RandomState(args.seed)
    sel = rng.choice(np.arange(N), size=n, replace=False).tolist()

    trigger = build_trigger(args)

    images = []
    for idx in sel:
        x, y = ds[idx]  # x may be PIL or Tensor
        # apply trigger but keep a copy of original too (optional)
        x_triggered = trigger.apply(x)
        # convert to tensor in CHW float format
        if isinstance(x_triggered, Image.Image):
            xt = TF.to_tensor(x_triggered)
        elif isinstance(x_triggered, torch.Tensor):
            xt = x_triggered.clone()
            # if tensor is uint8, convert to float
            if not torch.is_floating_point(xt):
                xt = xt.float() / 255.0
        else:
            # fallback: try to convert via TF
            xt = TF.to_tensor(x_triggered)

        # optionally undo normalization for visualization (if user provided mean/std)
        mean = parse_floats(args.undo_mean)
        std  = parse_floats(args.undo_std)
        if mean is not None and std is not None:
            xt = undo_normalize_tensor(xt, mean, std)

        xt = ensure_01_tensor(xt)
        images.append(xt)

    # make grid and save
    # compute grid rows/cols
    cols = int(np.ceil(np.sqrt(len(images))))
    grid = make_grid(images, nrow=cols, pad_value=1.0)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_image(grid, out_path)
    print(f"Saved preview image with {len(images)} triggered samples from client {client_id} to: {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--client", type=int, default=None, help="Client id to preview (if omitted, random client)")
    p.add_argument("--num", type=int, default=9, help="Number of images to preview")
    p.add_argument("--num-clients", type=int, default=8, dest="num_clients", help="Number of clients to partition into if adapter lacks get_client_loaders")
    p.add_argument("--batch", type=int, default=1, help="batch size used when building loaders (only used for loader API compatibility)")
    p.add_argument("--seed", type=int, default=0, help="seed for sampling images and random triggers")
    p.add_argument("--trigger", type=str, default="patch", choices=["patch","distributed","blended"], help="Trigger type")
    p.add_argument("--patch-size", type=int, default=3, help="patch size (for patch/distributed)")
    p.add_argument("--value", type=float, default=1.0, help="patch value for normalized tensors (use 255 for uint8)")
    p.add_argument("--num-patches", type=int, default=4, help="number of patches for distributed trigger")
    p.add_argument("--placement", type=str, default="corners", choices=["corners","grid","random"], help="placement for distributed trigger")
    p.add_argument("--alpha", type=float, default=0.12, help="alpha for blended trigger")
    p.add_argument("--pattern", type=str, default=None, help="path to pattern image for blended trigger")
    p.add_argument("--undo-mean", type=str, default=None, help="comma-separated mean to undo normalization for visualization e.g. 0.5,0.5,0.5")
    p.add_argument("--undo-std", type=str, default=None, help="comma-separated std to undo normalization for visualization e.g. 0.5,0.5,0.5")
    p.add_argument("--out", type=str, default="data/preview_trigger.png", help="where to save preview image")
    args = p.parse_args()
    preview_trigger_for_client(args)

if __name__ == "__main__":
    main()
