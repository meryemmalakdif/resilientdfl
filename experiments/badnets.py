#!/usr/bin/env python3
"""
experiments/badnets.py (updated)

Adds:
- attack scheduling via --attack-start and --attack-length
- per-round logging of ASR/clean acc
- durability metrics computed after attack stops

Run:
    PYTHONPATH=./src python experiments/badnets.py --clients 8 --rounds 8 --attack-start 2 --attack-length 3
"""
import argparse
import random
import time
import logging
from typing import Dict, Any, List, Optional, Callable
import csv
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

log = logging.getLogger("badnets")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---- project imports (unchanged) -------------------------------------------
from src.fl.baseserver import FedAvgAggregator
from src.fl.baseclient import BenignClient
from src.attacks.malicious_client import MaliciousClient

# components
try:
    from src.attacks.components import RandomSelector, PatchTrigger, NaivePoisoner
except Exception:
    raise ImportError("Could not import attack components from src.attacks.components")

# dataset adapter
from src.datasets.femnist import FEMNISTAdapter

# triggered helpers
from src.datasets.backdoor import make_triggered_loader, TriggeredTestset

# model - prefer lenet/mnist
try:
    from src.models.lenet import LeNet as ModelClass
except Exception:
    from src.models.mnist import MNIST as ModelClass


# ---- helper functions ------------------------------------------------------
def evaluate_clean(server: FedAvgAggregator, test_loader: DataLoader, device: torch.device):
    server.model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = server.model(x)
            loss_sum += loss_fn(out, y).item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total if total > 0 else float('nan')
    loss_avg = loss_sum / (len(test_loader) if len(test_loader) > 0 else 1.0)
    return acc, loss_avg


def evaluate_asr(server: FedAvgAggregator, test_dataset, trigger, target_label: int, device: torch.device, batch_size: int = 256):
    # uses make_triggered_loader to force the trigger on all samples
    loader = make_triggered_loader(base_dataset=test_dataset, trigger=trigger, keep_label=False, forced_label=target_label, fraction=1.0, seed=0, batch_size=batch_size, shuffle=False)
    server.model.eval()
    asr_count = 0
    total = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = server.model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            asr_count += int((preds == target_label).sum())
            total += preds.shape[0]
    return float(asr_count) / float(total) if total > 0 else float('nan')


def write_csv_log(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, "w", newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_durability(asr_history: List[float], attack_rounds: List[int]) -> Dict[str, Any]:
    """
    Compute durability metrics from per-round ASR history and attack round indices.
    asr_history: list indexed by round 0..R-1
    attack_rounds: list of round indices where attack was active (may be empty)
    """
    R = len(asr_history)
    if len(attack_rounds) == 0:
        return {
            "peak_asr": 0.0,
            "asr_at_attack_end": 0.0,
            "mean_post_asr": float('nan'),
            "normalized_auc_post": float('nan'),
            "half_life_rounds": None,
            "durability_score": float('nan'),
        }
    attack_start = min(attack_rounds)
    attack_end = max(attack_rounds)
    # peak during attack
    peak_asr = max(asr_history[r] for r in attack_rounds)
    asr_at_attack_end = asr_history[attack_end]
    # post-attack ASR sequence
    post_seq = asr_history[attack_end+1:] if attack_end+1 < R else []
    mean_post = float(np.mean(post_seq)) if len(post_seq) > 0 else float('nan')
    normalized_auc = mean_post  # ASR already normalized between 0 and 1
    # half-life: first round after attack_end when ASR <= peak/2
    half_life = None
    target_half = peak_asr / 2.0 if peak_asr > 0 else 0.0
    for i, val in enumerate(post_seq):
        if val <= target_half:
            half_life = i + 1  # rounds after attack_end
            break
    # durability score: mean_post / peak_asr (how much of peak is retained on average)
    durability_score = (mean_post / peak_asr) if (peak_asr > 0 and not math.isnan(mean_post)) else float('nan')
    return {
        "peak_asr": float(peak_asr),
        "asr_at_attack_end": float(asr_at_attack_end),
        "mean_post_asr": float(mean_post) if not math.isnan(mean_post) else float('nan'),
        "normalized_auc_post": float(normalized_auc) if not math.isnan(normalized_auc) else float('nan'),
        "half_life_rounds": half_life,
        "durability_score": float(durability_score) if not math.isnan(durability_score) else float('nan'),
    }


# ---- experiment -----------------------------------------------------------
def run_badnets(
    num_clients: int = 8,
    rounds: int = 6,
    local_epochs: int = 1,
    local_batch: int = 32,
    lr: float = 0.05,
    weight_decay: float = 0.0,
    fraction_malicious: float = 0.25,
    poison_frac: float = 0.1,
    target_label: int = 0,
    seed: int = 123,
    device_str: str = None,
    attack_start: int = 0,
    attack_length: int = 1,
    csv_out: Optional[str] = "badnets_log.csv",
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str if device_str is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Device: %s", device)

    # --- dataset adapter and loaders (FEMNISTAdapter expected to exist) -----
    try:
        train_adapter = FEMNISTAdapter(root="data", train=True, download=False)
        test_adapter = FEMNISTAdapter(root="data", train=False, download=False)
    except TypeError:
        train_adapter = FEMNISTAdapter()
        test_adapter = FEMNISTAdapter(train=False)

    # client loaders: uses adapter.get_client_loaders or partitions adapter.dataset
    if hasattr(train_adapter, "get_client_loaders"):
        client_loaders = train_adapter.get_client_loaders(num_clients=num_clients, strategy="iid", batch_size=local_batch)
    else:
        ds = train_adapter.dataset
        idx = np.arange(len(ds)); np.random.shuffle(idx)
        splits = np.array_split(idx, num_clients)
        client_loaders = {i: DataLoader(torch.utils.data.Subset(ds, list(map(int, splits[i]))), batch_size=local_batch, shuffle=True, num_workers=2) for i in range(num_clients)}

    # test loader & dataset
    if hasattr(test_adapter, "get_test_loader"):
        test_loader = test_adapter.get_test_loader(batch_size=256)
        test_dataset = test_adapter.dataset
    else:
        test_loader = DataLoader(test_adapter.dataset, batch_size=256, shuffle=False, num_workers=2)
        test_dataset = test_adapter.dataset

    # assemble clients
    clients = {}
    num_mal = max(1, int(round(fraction_malicious * num_clients)))
    mal_ids = set(sorted(random.sample(list(range(num_clients)), num_mal)))
    log.info("Malicious clients: %s", mal_ids)

    for cid in range(num_clients):
        model = ModelClass()
        if cid in mal_ids:
            selector = RandomSelector(seed=seed + cid)
            trigger = PatchTrigger(patch_size=3, value=1.0)
            poisoner = None  # naive path via MaliciousClient
            client = MaliciousClient(
                id=cid, trainloader=client_loaders[cid], testloader=None, model=model,
                lr=lr, weight_decay=weight_decay, epochs=local_epochs, device=device,
                selector=selector, trigger=trigger, poisoner=poisoner,
                target_label=target_label, poison_fraction=poison_frac
            )
        else:
            client = BenignClient(id=cid, trainloader=client_loaders[cid], testloader=None, model=model,
                                  lr=lr, weight_decay=weight_decay, epochs=local_epochs, device=device)
        clients[cid] = client

    # server
    global_model = ModelClass()
    server = FedAvgAggregator(model=global_model, testloader=test_loader, device=device)

    # attack rounds set
    attack_rounds = list(range(attack_start, min(rounds, attack_start + attack_length)))
    log.info("Attack active in rounds: %s", attack_rounds)

    # We'll log per-round metrics for CSV + durability computation
    rows = []
    asr_history: List[float] = []

    # Main FL loop
    for r in range(rounds):
        t0 = time.time()
        log.info("=== Round %d/%d ===", r + 1, rounds)
        global_params = server.get_params()

        selected_client_ids = sorted(clients.keys())
        for cid in selected_client_ids:
            c = clients[cid]
            c.set_params(global_params)

            # If this client is malicious and attack is NOT active this round:
            # temporarily disable its poisoning by monkey-patching _select_poison_indices to return [].
            disabled = False
            orig_select = None
            if cid in mal_ids and r not in attack_rounds:
                # disable
                if hasattr(c, "_select_poison_indices"):
                    orig_select = c._select_poison_indices
                    c._select_poison_indices = (lambda num_poison=None: [])  # type: ignore
                    disabled = True

            # local update
            update = c.local_train(epochs=local_epochs, round_idx=r)

            # restore original selector method if we disabled it
            if disabled and orig_select is not None:
                c._select_poison_indices = orig_select  # type: ignore

            server.receive_update(update["weights"], update["num_samples"])

        # aggregate
        server.aggregate()

        # evaluate clean acc & ASR
        clean_acc, clean_loss = evaluate_clean(server, test_loader, device)
        asr = evaluate_asr(server, test_dataset, trigger, target_label=target_label, device=device)

        asr_history.append(asr)
        row = {
            "round": r,
            "time_s": round(time.time() - t0, 3),
            "clean_acc": clean_acc,
            "clean_loss": clean_loss,
            "asr": asr,
            "attack_active": (r in attack_rounds)
        }
        rows.append(row)
        log.info("Round %d: clean_acc=%.4f loss=%.4f ASR=%.4f attack_active=%s", r, clean_acc, clean_loss, asr, row["attack_active"])

    # Save CSV log if requested
    if csv_out:
        fieldnames = ["round", "time_s", "clean_acc", "clean_loss", "asr", "attack_active"]
        write_csv_log(csv_out, rows, fieldnames)
        log.info("Wrote per-round metrics to %s", csv_out)

    # compute durability metrics
    dur = compute_durability(asr_history, attack_rounds)
    log.info("Durability metrics: %s", dur)

    # save final model
    out_path = "badnets_final_global.pth"
    torch.save(server.model.state_dict(), out_path)
    log.info("Saved final global model to %s", out_path)


# ---- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=8)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--local-batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fraction-malicious", type=float, default=0.25)
    p.add_argument("--poison-frac", type=float, default=0.1)
    p.add_argument("--target-label", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--attack-start", type=int, default=0, help="Round index where attack begins (0-indexed)")
    p.add_argument("--attack-length", type=int, default=1, help="Number of consecutive rounds the attack is active")
    p.add_argument("--csv-out", type=str, default="badnets_log.csv", help="Path to write per-round metrics CSV (set empty to skip)")
    args = p.parse_args()

    run_badnets(
        num_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_batch=args.local_batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fraction_malicious=args.fraction_malicious,
        poison_frac=args.poison_frac,
        target_label=args.target_label,
        seed=args.seed,
        device_str=args.device,
        attack_start=args.attack_start,
        attack_length=args.attack_length,
        csv_out=args.csv_out if args.csv_out else None,
    )
