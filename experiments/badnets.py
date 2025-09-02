"""
BadNets-style federated learning experiment (FEMNIST).
- Random selection of local samples to poison
- Naive (data-level) poisoning using a patch trigger
- Malicious clients send poisoned updates via their usual local_train API

Usage:
    PYTHONPATH=./src python experiments/badnets.py --clients 10 --rounds 5
"""
import argparse
import random
import time
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# ---- Project imports --------------------------------
from src.fl.baseclient import BenignClient
from src.fl.baseserver import FedAvgAggregator

from src.attacks.selectors import RandomSelector
from src.attacks.triggers import PatchTrigger
from src.attacks.poisoners import NaiveDataPoisoner
try:
    from src.attacks.malicious_client import MaliciousClient
except Exception:
    MaliciousClient = None  # fallback defined later

# FEMNIST adapter (you said wrapper implemented already)
DatasetAdapterClass = None
try:
    from src.datasets.femnist import FEMNISTAdapter
    DatasetAdapterClass = FEMNISTAdapter
except Exception:
    # try generic adapter
    try:
        from src.datasets.adapter import DatasetAdapter
        DatasetAdapterClass = DatasetAdapter
    except Exception:
        DatasetAdapterClass = None

# backdoor/testset helper
try:
    from src.datasets.backdoor import TriggeredTestset
except Exception:
    TriggeredTestset = None

# model for FEMNIST - assume your models.mnist.MNIST is suitable
try:
    from src.models.mnist import MNIST as MNISTModel
except Exception:
    # fallback tiny model
    import torch.nn as nn
    class MNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(28*28, 10)
        def forward(self, x):
            x = self.flatten(x)
            return self.fc(x)

# ---- Helpers ---------------------------------------------------------------
def make_client_loaders_from_femnist_adapter(adapter, num_clients: int, batch_size: int):
    """
    Use adapter.get_client_loaders(...) if present; otherwise partition IID and return DataLoaders.
    """
    if hasattr(adapter, "get_client_loaders"):
        return adapter.get_client_loaders(num_clients=num_clients, strategy="iid", batch_size=batch_size)
    # attempt to use adapter.dataset => split
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
    raise RuntimeError("Adapter does not expose get_client_loaders or dataset.")

def compute_clean_accuracy(server: FedAvgAggregator, test_loader: DataLoader, device: torch.device):
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
    acc = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / (len(test_loader) if len(test_loader)>0 else 1.0)
    return acc, avg_loss

def compute_asr(server: FedAvgAggregator, test_dataset, trigger, target_label: int, device: torch.device, batch_size: int = 256):
    """
    Compute Attack Success Rate (ASR) using TriggeredTestset wrapper (if available).
    If TriggeredTestset is not available, fall back to applying trigger per-batch (expects trigger.apply supports tensors).
    """
    server.model.eval()
    # prefer TriggeredTestset helper
    if TriggeredTestset is not None:
        trig_ds = TriggeredTestset(test_dataset, trigger, keep_label=False, forced_label=None)
        loader = DataLoader(trig_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        asr_count = 0
        total = 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                out = server.model(x)
                preds = out.argmax(dim=1).cpu().numpy()
                asr_count += int((preds == target_label).sum())
                total += preds.shape[0]
        return float(asr_count) / float(total) if total > 0 else 0.0

    # fallback: apply trigger manually per batch (works if trigger.apply accepts tensors)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    asr_count = 0
    total = 0
    with torch.no_grad():
        for x, _ in loader:
            # apply trigger to each sample in the batch
            if isinstance(x, torch.Tensor):
                trig_batch = []
                for i in range(x.size(0)):
                    xi = x[i]
                    xt = trigger.apply(xi)
                    if not isinstance(xt, torch.Tensor):
                        from torchvision.transforms.functional import to_tensor
                        xt = to_tensor(xt)
                    trig_batch.append(xt)
                xt = torch.stack(trig_batch, dim=0).to(device)
            else:
                from torchvision.transforms.functional import to_tensor
                trig_batch = []
                for i in range(len(x)):
                    xt = trigger.apply(x[i])
                    if not isinstance(xt, torch.Tensor):
                        xt = to_tensor(xt)
                    trig_batch.append(xt)
                xt = torch.stack(trig_batch, dim=0).to(device)
            out = server.model(xt)
            preds = out.argmax(dim=1).cpu().numpy()
            asr_count += int((preds == target_label).sum())
            total += preds.shape[0]
    return float(asr_count) / float(total) if total > 0 else 0.0

# ---- Fallback malicious client if your MaliciousClient is missing ----------------
if MaliciousClient is None:
    class MaliciousClient(BenignClient):
        def __init__(self, *args, poisoner=None, selector=None, trigger=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._poisoner = poisoner
            self._selector = selector
            self._trigger = trigger
        def local_train(self, epochs: int, round_idx: int):
            # delegate to poisoner (it should return the usual update dict)
            return self._poisoner.poison_and_train(self, self._selector, self._trigger, epochs=epochs, round_idx=round_idx, global_params=self.get_params())

# ---- Main experiment -------------------------------------------------------
def run_badnets(
    num_clients: int = 10,
    rounds: int = 5,
    local_epochs: int = 1,
    local_batch: int = 32,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    fraction_malicious: float = 0.2,
    poison_frac: float = 0.1,
    seed: int = 123,
    device_str: str = None,
):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device_str if device_str is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    # ---- load FEMNIST adapter and create client loaders --------------------------------
    if DatasetAdapterClass is None:
        raise RuntimeError("FEMNIST adapter not found at src.datasets.femnist. Please ensure it exists.")

    # instantiate train & test adapters (constructor signatures may vary)
    try:
        train_adapter = DatasetAdapterClass(root="data", train=True, download=False)
        test_adapter = DatasetAdapterClass(root="data", train=False, download=False)
    except TypeError:
        # fallback: try without args
        train_adapter = DatasetAdapterClass()
        test_adapter = DatasetAdapterClass(train=False)

    client_loaders = make_client_loaders_from_femnist_adapter(train_adapter, num_clients=num_clients, batch_size=local_batch)
    # test loader: prefer adapter.get_test_loader or get_test_loader name
    if hasattr(test_adapter, "get_test_loader"):
        test_loader = test_adapter.get_test_loader(batch_size=256)
        test_dataset = test_adapter.dataset if hasattr(test_adapter, "dataset") else None
    elif hasattr(test_adapter, "get_dataloader"):
        test_loader = test_adapter.get_dataloader(batch_size=256)
        test_dataset = test_adapter.dataset if hasattr(test_adapter, "dataset") else None
    else:
        # try to get DataLoader from loader util
        from src.datasets.loaders import get_dataloader
        test_loader = get_dataloader("mnist", batch_size=256, shuffle=False, train=False, download=False)
        test_dataset = test_loader.dataset

    # ---- build clients --------------------------------------------------------
    clients: Dict[int, Any] = {}
    malicious_count = max(1, int(round(fraction_malicious * num_clients)))
    malicious_ids = set(sorted(random.sample(list(range(num_clients)), malicious_count)))
    print("Malicious clients:", malicious_ids)

    for cid in range(num_clients):
        model = MNISTModel()
        if cid in malicious_ids:
            selector = RandomSelector(seed=seed + cid)
            trigger = PatchTrigger(patch_size=3, value=255)  # works with tensor/PIL
            poisoner = NaiveDataPoisoner(target_label=0, poison_fraction=poison_frac)

            client = MaliciousClient(
                id=cid,
                trainloader=client_loaders[cid],
                testloader=None,
                model=model,
                lr=lr,
                weight_decay=weight_decay,
                epochs=local_epochs,
                device=device,
                poisoner=poisoner,
                selector=selector,
                trigger=trigger
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

    # ---- server ---------------------------------------------------------------
    global_model = MNISTModel()
    server = FedAvgAggregator(model=global_model, testloader=test_loader, device=device)

    # ---- federated rounds ----------------------------------------------------
    print("Starting FL rounds...")
    for r in range(rounds):
        t0 = time.time()
        print(f"\n=== Round {r+1}/{rounds} ===")
        global_params = server.get_params()

        # select clients (here all clients for simplicity)
        selected = list(clients.keys())

        for cid in selected:
            c = clients[cid]
            # broadcast global params
            c.set_params(global_params)
            # each client's local_train returns an update dict
            update = c.local_train(epochs=local_epochs, round_idx=r)
            server.receive_update(update["weights"], update["num_samples"])

        # aggregate
        server.aggregate()

        # evaluate clean accuracy & ASR
        clean_acc, clean_loss = compute_clean_accuracy(server, test_loader, device)
        # compute ASR using trigger of malicious client (target_label=0)
        sample_trigger = PatchTrigger(patch_size=3, value=255)
        # prefer passing raw test dataset to compute_asr; try to get underlying dataset
        test_ds = None
        if hasattr(test_adapter, "dataset"):
            test_ds = test_adapter.dataset
        elif hasattr(test_loader, "dataset"):
            test_ds = test_loader.dataset
        asr = compute_asr(server, test_ds, sample_trigger, target_label=0, device=device) if test_ds is not None else float('nan')

        print(f"Round {r+1} done [{time.time()-t0:.1f}s]  Clean ACC={clean_acc:.4f}, Loss={clean_loss:.4f}, ASR={asr:.4f}")

    # save final model
    torch.save(server.model.state_dict(), "badnets_final_global.pth")
    print("Saved badnets_final_global.pth")


# ---- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--local-batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--fraction-malicious", type=float, default=0.2)
    parser.add_argument("--poison-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_badnets(
        num_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_batch=args.local_batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fraction_malicious=args.fraction_malicious,
        poison_frac=args.poison_frac,
        seed=args.seed,
        device_str=args.device,
    )
