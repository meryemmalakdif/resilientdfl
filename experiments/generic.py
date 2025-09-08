#!/usr/bin/env python3
"""
run_generic_experiment.py

A generic, CLI-configurable script for running federated learning simulations
with various datasets, attacks, and (eventually) defenses.
"""

import random
import copy
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# --- Import Framework Components ---
from src.fl.baseclient import BenignClient
from src.fl.baseserver import FedAvgAggregator

# Import Dataset Adapters
from src.datasets.femnist import FEMNISTAdapter
# (Assuming you will create similar adapters for other datasets)
# from src.datasets.cifar10 import CIFAR10Adapter
# from src.datasets.mnist import MNISTAdapter
# from src.datasets.gtsrb import GTSRBAdapter

# Import Models
from src.models.lenet import LeNet5
from src.models.simple_cifar import SimpleCNN_CIFAR
from src.models.gtsrb_cnn import GTSRB_CNN

# Import Backdoor Utilities
from src.datasets.backdoor import create_asr_test_loader

# Import Attack Clients and Components
from src.attacks.badnets_client import BadNetsClient
from src.attacks.scaling_client import ScalingAttackClient
from src.attacks.neurotoxin_client import NeurotoxinClient
# --- MODIFICATION START: Import A3FL components ---
from src.attacks.a3fl_client import A3FLClient
from src.attacks.triggers.a3fl import A3FLTrigger
# --- MODIFICATION END ---
from src.attacks.triggers.patch import PatchTrigger
from src.attacks.selectors.randomselector import RandomSelector

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_asr(model: torch.nn.Module, test_dataset: Dataset, trigger, target_label: int, device: torch.device, batch_size: int = 256):
    """Evaluates the Attack Success Rate (ASR) of the model."""
    backdoor_loader = create_asr_test_loader(
        base_dataset=test_dataset,
        trigger=trigger,
        target_class=target_label,
        batch_size=batch_size
    )
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in backdoor_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return (correct / total) if total > 0 else 0.0

def build_clients(client_loaders, test_loader, model_cls, args, selector, trigger):
    """Builds a mix of benign and malicious clients based on CLI args."""
    clients = []
    for cid, loader in client_loaders.items():
        client_kwargs = {
            'id': cid,
            'trainloader': loader,
            'testloader': test_loader,
            'model': model_cls().to(args.device),
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'epochs': args.local_epochs,
            'device': args.device
        }
        if cid < args.num_malicious:
            if args.attack == 'badnets':
                client = BadNetsClient(**client_kwargs, selector=selector, trigger=trigger, target_class=args.target_class)
            elif args.attack == 'scaling':
                client = ScalingAttackClient(**client_kwargs, selector=selector, trigger=trigger, target_class=args.target_class,
                                             attack_round=args.attack_round, scale_factor=args.scale_factor, num_clients=args.num_clients,
                                             num_malicious=args.num_malicious)
            elif args.attack == 'neurotoxin':
                client = NeurotoxinClient(**client_kwargs, selector=selector, trigger=trigger, target_class=args.target_class,
                                          attack_start_round=args.attack_round, mask_k_percent=args.mask_k_percent,
                                          malicious_epochs=args.malicious_epochs)
            # --- MODIFICATION START: Add A3FL client case ---
            elif args.attack == 'a3fl':
                if not isinstance(trigger, A3FLTrigger):
                    raise ValueError("A3FL attack requires an A3FLTrigger instance.")
                client = A3FLClient(**client_kwargs, selector=selector, trigger=trigger, target_class=args.target_class,
                                      trigger_epochs=args.trigger_epochs, trigger_lr=args.trigger_lr)
            # --- MODIFICATION END ---
            else:
                raise ValueError(f"Unknown attack type: {args.attack}")
        else:
            client = BenignClient(**client_kwargs)
        clients.append(client)
    return clients

def main():
    parser = argparse.ArgumentParser(description="Generic Federated Learning Experiment Runner")
    # --- FL Parameters ---
    parser.add_argument("--dataset", type=str, required=True, choices=["femnist", "cifar10", "mnist", "gtsrb"])
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    # --- Attack Parameters ---
    # --- MODIFICATION START: Add a3fl as a choice ---
    parser.add_argument("--attack", type=str, default="none", choices=["none", "badnets", "scaling", "neurotoxin", "a3fl"])
    # --- MODIFICATION END ---
    parser.add_argument("--num_malicious", type=int, default=0)
    parser.add_argument("--target_class", type=int, default=5)
    parser.add_argument("--poisoning_rate", type=float, default=0.3, help="Fraction of data to poison on malicious clients")
    parser.add_argument("--attack_round", type=int, default=15, help="Round to start the attack (for scaling/neurotoxin)")
    parser.add_argument("--scale_factor", type=float, default=0.0, help="Scale factor for model scaling attack (0 for dynamic)")
    parser.add_argument("--mask_k_percent", type=float, default=0.05, help="Top-k% of gradients to mask for Neurotoxin")
    parser.add_argument("--malicious_epochs", type=int, default=5, help="Number of local epochs for malicious clients (to boost signal)")
    # --- MODIFICATION START: Add A3FL specific arguments ---
    parser.add_argument("--trigger_epochs", type=int, default=5, help="Epochs for A3FL trigger training per round")
    parser.add_argument("--trigger_lr", type=float, default=0.01, help="Learning rate for A3FL trigger training")
    # --- MODIFICATION END ---
    args = parser.parse_args()

    # --- Setup ---
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {args.device}")

    # --- Dataset and Model Selection ---
    if args.dataset == 'femnist':
        dataset_adapter = FEMNISTAdapter(root="data", train=True, download=True)
        model_cls = lambda: LeNet5(num_classes=62)
        # --- MODIFICATION START: Select trigger based on attack type ---
        if args.attack == 'a3fl':
            trigger = A3FLTrigger(size=(3, 3))
        else:
            trigger = PatchTrigger(position=(25, 25), size=(3, 3), color=(2.0,))
        # --- MODIFICATION END ---
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not implemented yet.")

    client_loaders = dataset_adapter.get_client_loaders(num_clients=args.num_clients, strategy="iid", batch_size=args.batch_size, seed=args.seed)
    test_loader = dataset_adapter.get_test_loader(batch_size=args.batch_size)
    
    # --- Server and Client Initialization ---
    server = FedAvgAggregator(model=model_cls().to(args.device), testloader=test_loader, device=args.device)
    selector = RandomSelector(poisoning_rate=args.poisoning_rate)
    clients = build_clients(client_loaders, test_loader, model_cls, args, selector, trigger)

    # --- Main FL Loop ---
    prev_model_params = None
    for round_idx in range(args.num_rounds):
        print(f"\n--- Round {round_idx + 1}/{args.num_rounds} ---")
        
        model_params = server.get_params()
        
        agg_grad = None
        if prev_model_params:
            agg_grad = {name: model_params[name].to(args.device) - prev_model_params[name].to(args.device) for name in model_params}
        prev_model_params = copy.deepcopy(model_params)

        for client in clients:
            client.set_params(model_params)
            
            extra_args = {}
            if isinstance(client, (ScalingAttackClient, NeurotoxinClient)):
                extra_args['prev_global_grad'] = agg_grad

            update = client.local_train(epochs=args.local_epochs, round_idx=round_idx, **extra_args)
            server.receive_update(update['weights'], length=update['num_samples'])
        
        server.aggregate()
        
        # --- Evaluation ---
        main_metrics = server.evaluate()
        print(f"Round {round_idx + 1}: Main Accuracy = {main_metrics['metrics']['main_accuracy']:.4f}")
        
        if args.num_malicious > 0:
            asr = evaluate_asr(server.model, test_loader.dataset, trigger, args.target_class, args.device)
            print(f"Round {round_idx + 1}: Backdoor Accuracy = {asr:.4f}")

    print("\n--- Training Finished ---")
    final_main_metrics = server.evaluate()
    print(f"Final Main Accuracy: {final_main_metrics['metrics']['main_accuracy']:.4f}")
    if args.num_malicious > 0:
        final_asr = evaluate_asr(server.model, test_loader.dataset, trigger, args.target_class, args.device)
        print(f"Final Backdoor Accuracy (ASR): {final_asr:.4f}")

if __name__ == "__main__":
    main()

