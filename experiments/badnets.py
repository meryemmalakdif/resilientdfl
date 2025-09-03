#!/usr/bin/env python3
"""
experiments/badnets.py

BadNets experiment using your Selector->Trigger->Poisoner decomposition and MaliciousClient.

Usage:
    PYTHONPATH=./src python experiments/badnets.py --clients 8 --rounds 6 --poison-frac 0.1
"""
import argparse
import random
import time
import logging
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

log = logging.getLogger("badnets")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---- Project imports (expect these modules to exist) -------------------------
from src.fl.baseserver import FedAvgAggregator
from src.fl.baseclient import BenignClient

from src.attacks.malicious_client import MaliciousClient

# Try to import component classes from unified components; fall back to previous modules if needed
try:
    from src.attacks.components import RandomSelector, PatchTrigger, NaivePoisoner
except Exception:
    raise ImportError("Could not import attack components from src.attacks.components")


# FEMNIST adapter (your wrapper)
try:
    from src.datasets.femnist import FEMNISTAdapter
    HAVE_FEMNIST = True
except Exception:
    FEMNISTAdapter = None
    HAVE_FEMNIST = False

# Triggered testset helper
try:
    from src.datasets.backdoor import make_triggered_loader, TriggeredTestset
    HAVE_TRIGGER_HELPERS = True
except Exception:
    make_triggered_loader = None
    TriggeredTestset = None
    HAVE_TRIGGER_HELPERS = False

# model: prefer lenet or mnist model
try:
    from src.models.lenet import LeNet as ModelClass
except Exception:
    try:
        from src.models.mnist import MNIST as ModelClass
    except Exception:
        # fallback simple linear
        import torch.nn as nn
        class ModelClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(28*28, 10)
            def forward(self, x):
                return self.fc(self.flatten(x))


# ---- helpers ---------------------------------------------------------------
def build_client_loaders(adapter, num_clients: int, batch_size: int):
    """
    Prefer adapter.get_client_loaders(num_clients, strategy, batch_size).
    If unavailable, partition adapter.dataset IID.
    Returns dict: client_id -> DataLoader
    """
    if adapter is None:
        raise RuntimeError("No dataset adapter available")
    if hasattr(adapter, "get_client_loaders"):
        try:
            return adapter.get_client_loaders(num_clients=num_clients, strategy="iid", batch_size=batch_size)
        except TypeError:
            # some signatures might differ: try without keywords
            return adapter.get_client_loaders(num_clients, "iid", batch_size)
    # fallback: partition adapter.dataset into equal splits
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
    raise RuntimeError("Adapter has neither get_client_loaders nor dataset attribute.")


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
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / (len(test_loader) if len(test_loader) > 0 else 1.0)
    return acc, avg_loss


def evaluate_asr(server: FedAvgAggregator, test_dataset, trigger, target_label: int, device: torch.device, batch_size: int = 256):
    server.model.eval()
    # try TriggeredTestset helper
    if make_triggered_loader is not None:
        loader = make_triggered_loader(base_dataset=test_dataset, trigger=trigger, keep_label=False, forced_label=target_label, fraction=1.0, seed=0, batch_size=batch_size, shuffle=False)
    else:
        # fallback: create on-the-fly loader that applies trigger per-batch (trigger.apply must accept tensors)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    asr_count = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            # if using fallback loader we must trigger manually
            if make_triggered_loader is None:
                # per-sample trigger (supports tensor or PIL)
                if isinstance(x, torch.Tensor):
                    triggered = []
                    for i in range(x.size(0)):
                        xi = x[i]
                        xt = trigger.apply(xi)
                        if not isinstance(xt, torch.Tensor):
                            from torchvision.transforms.functional import to_tensor
                            xt = to_tensor(xt)
                        triggered.append(xt)
                    xt = torch.stack(triggered, dim=0).to(device)
                else:
                    from torchvision.transforms.functional import to_tensor
                    triggered = []
                    for i in range(len(x)):
                        xt = trigger.apply(x[i])
                        if not isinstance(xt, torch.Tensor):
                            xt = to_tensor(xt)
                        triggered.append(xt)
                    xt = torch.stack(triggered, dim=0).to(device)
            else:
                xt = x.to(device)
            out = server.model(xt)
            preds = out.argmax(dim=1).cpu().numpy()
            asr_count += int((preds == target_label).sum())
            total += preds.shape[0]
    return float(asr_count) / float(total) if total > 0 else 0.0


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
):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str if device_str is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Device: %s", device)

    # ---- dataset & client loaders ------------------------------------------
    if HAVE_FEMNIST and FEMNISTAdapter is not None:
        # try common constructor signatures
        try:
            train_adapter = FEMNISTAdapter(root="data", train=True, download=False)
            test_adapter = FEMNISTAdapter(root="data", train=False, download=False)
        except TypeError:
            train_adapter = FEMNISTAdapter()
            test_adapter = FEMNISTAdapter(train=False)
        client_loaders = build_client_loaders(train_adapter, num_clients=num_clients, batch_size=local_batch)
        # get test loader/dataset
        if hasattr(test_adapter, "get_test_loader"):
            test_loader = test_adapter.get_test_loader(batch_size=256)
            test_dataset = test_adapter.dataset if hasattr(test_adapter, "dataset") else test_loader.dataset
        elif hasattr(test_adapter, "dataset"):
            test_dataset = test_adapter.dataset
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
        else:
            raise RuntimeError("FEMNIST adapter lacks test loader / dataset")
    else:
        # fallback: if adapter missing, error because you said dataset class provides loaders
        raise RuntimeError("FEMNIST adapter not available. Place your FEMNIST adapter at src.datasets.femnist or adapt this script.")

    # ---- build clients -----------------------------------------------------
    clients: Dict[int, Any] = {}
    num_mal = max(1, int(round(fraction_malicious * num_clients)))
    mal_ids = set(sorted(random.sample(list(range(num_clients)), num_mal)))
    log.info("Malicious clients: %s", mal_ids)

    for cid in range(num_clients):
        model = ModelClass()
        if cid in mal_ids:
            selector = RandomSelector(seed=seed + cid) if RandomSelector is not None else None
            trigger = PatchTrigger(patch_size=3, value=1.0) if PatchTrigger is not None else None
            # poisoner: None so MaliciousClient uses naive PoisonedWrapper (training on poisoned+clean)
            poisoner = None
            client = MaliciousClient(
                id=cid,
                trainloader=client_loaders[cid],
                testloader=None,
                model=model,
                lr=lr,
                weight_decay=weight_decay,
                epochs=local_epochs,
                device=device,
                selector=selector,
                trigger=trigger,
                poisoner=poisoner,
                target_label=target_label,
                poison_fraction=poison_frac
            )
        else:
            client = BenignClient(
                id=cid,
                trainloader=client_loaders[cid],
                testloader=None,
                model=model,
                lr=lr,
                weight_decay=weight_decay,
                epochs=local_epochs,
                device=device
            )
        clients[cid] = client

    # ---- server -------------------------------------------------------------
    global_model = ModelClass()
    server = FedAvgAggregator(model=global_model, testloader=test_loader, device=device)

    # ---- federated rounds --------------------------------------------------
    log.info("Starting training: %d rounds", rounds)
    for r in range(rounds):
        t0 = time.time()
        log.info("=== Round %d/%d ===", r + 1, rounds)
        global_params = server.get_params()

        # broadcast + local training
        for cid in sorted(clients.keys()):
            c = clients[cid]
            c.set_params(global_params)
            update = c.local_train(epochs=local_epochs, round_idx=r)
            server.receive_update(update["weights"], update["num_samples"])

        # aggregate
        server.aggregate()

        # evaluate
        clean_acc, clean_loss = evaluate_clean(server, test_loader, device)
        asr = evaluate_asr(server, test_dataset, trigger, target_label=target_label, device=device)

        log.info("Round %d done (%.1fs)  Clean ACC=%.4f  Loss=%.4f  ASR=%.4f", r + 1, time.time() - t0, clean_acc, clean_loss, asr)

    # save final model
    out_path = "badnets_final_global.pth"
    torch.save(server.model.state_dict(), out_path)
    log.info("Saved global model to %s", out_path)


# ---- CLI -----------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=8)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--local-batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fraction-malicious", type=float, default=0.25)
    p.add_argument("--poison-frac", type=float, default=0.1)
    p.add_argument("--target-label", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default=None)
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
    )
