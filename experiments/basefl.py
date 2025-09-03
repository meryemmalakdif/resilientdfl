#!/usr/bin/env python3
"""
run_femnist_fl.py

Run a simple FedAvg loop on FEMNIST with CLI-configurable parameters.
"""

import argparse
import copy
import random
from typing import Dict, Any

import numpy as np
import torch

from src.fl.baseclient import BenignClient
from src.fl.baseserver import FedAvgAggregator
from src.datasets.femnist import FEMNISTAdapter
from src.datasets.mnist import MNISTAdapter
from src.models.lenet import LeNet5
from src.models.alexnet import AlexNet_FMNIST
from src.models.mnist import MNIST

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_clients(client_loaders: Dict[Any, torch.utils.data.DataLoader],
                  test_loader: torch.utils.data.DataLoader,
                  model_cls,
                  lr: float,
                  weight_decay: float,
                  epochs: int,
                  device: torch.device):
    clients = []
    for cid, loader in client_loaders.items():
        # create a fresh model instance per client to avoid shared references
        client_model = model_cls(num_classes=62).to(device)
        client = BenignClient(
            id=cid,
            trainloader=loader,
            testloader=test_loader,
            model=client_model,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            device=device
        )
        clients.append(client)
    return clients


def main():
    parser = argparse.ArgumentParser(description="Federated FEMNIST with FedAvg (CLI-configurable)")
    parser.add_argument("--data-root", type=str, default="data", help="dataset root")
    parser.add_argument("--num-clients", type=int, default=5, help="number of clients to simulate")
    parser.add_argument("--num-rounds", type=int, default=5, help="number of federated rounds")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size per client")
    parser.add_argument("--lr", type=float, default=0.01, help="client learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for optimizer")
    parser.add_argument("--epochs", type=int, default=1, help="local epochs per round")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--strategy", type=str, default="iid", choices=["iid", "non-iid"], help="client split strategy")
    parser.add_argument("--download", action="store_true", help="download dataset if missing")
    args = parser.parse_args()

    # set seed & device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, seed: {args.seed}")

    # prepare dataset
    datasetAdapter = FEMNISTAdapter(root=args.data_root, train=True, download=args.download)
    # datasetAdapter.dataset not used directly here, but kept for compatibility if needed
    data = datasetAdapter.dataset

    client_loaders = datasetAdapter.get_client_loaders(
        num_clients=args.num_clients,
        strategy=args.strategy,
        batch_size=args.batch_size,
        seed=args.seed
    )
    test_loader = datasetAdapter.get_test_loader(batch_size=args.batch_size)

    # create server global model instance
    global_model = LeNet5(num_classes=62).to(device)
    server = FedAvgAggregator(model=global_model, testloader=test_loader, device=device)

    # create clients (each with its own model instance)
    clients = build_clients(
        
        client_loaders=client_loaders,
        test_loader=test_loader,
        model_cls=LeNet5,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=device
    )

    # federated training loop
    for round_idx in range(args.num_rounds):
        print(f"\n--- Round {round_idx + 1}/{args.num_rounds} ---")

        # get global params and distribute to clients
        model_params = server.get_params()
        for client in clients:
            client.set_params(model_params)

        # each client trains locally and sends updates
        for client in clients:
            update = client.local_train(epochs=args.epochs, round_idx=round_idx)
            server.receive_update(update['weights'], length=update['num_samples'])

        # aggregate and evaluate on server
        aggregated_weights = server.aggregate()
        metrics = server.evaluate()
        print(f"Server evaluation (round {round_idx+1}): {metrics}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()

# PYTHONPATH=. python experiments/basefl.py --num-clients 5 --num-rounds 10 --batch-size 64 --lr 0.05 --seed 42 --strategy iid
