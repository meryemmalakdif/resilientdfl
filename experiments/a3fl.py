#!/usr/bin/env python3
"""
run_a3fl_mnist.py

A simple, focused experiment to test the A3FL backdoor attack on MNIST.
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# --- Import Framework Components ---
from src.fl.baseclient import BenignClient
from src.fl.baseserver import FedAvgAggregator
from src.datasets.mnist import MNISTAdapter
from src.models.lenet import LeNet5
from src.datasets.backdoor import create_asr_test_loader

# --- Import A3FL Components ---
from src.attacks.a3fl_client import A3FLClient
from src.attacks.triggers.a3fl import A3FLTrigger
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

def build_clients(client_loaders, test_loader, model_cls, config, selector, trigger):
    """Builds a mix of benign and A3FL clients."""
    clients = []
    for cid, loader in client_loaders.items():
        client_kwargs = {
            'id': cid,
            'trainloader': loader,
            'testloader': test_loader,
            'model': model_cls().to(config['DEVICE']),
            'lr': config['LEARNING_RATE'],
            'weight_decay': 0.0,
            'epochs': config['LOCAL_EPOCHS'],
            'device': config['DEVICE']
        }
        if cid < config['NUM_MALICIOUS']:
            client = A3FLClient(
                **client_kwargs,
                selector=selector,
                trigger=trigger,
                target_class=config['TARGET_CLASS'],
                trigger_epochs=config['TRIGGER_EPOCHS'],
                trigger_lr=config['TRIGGER_LR']
            )
        else:
            client = BenignClient(**client_kwargs)
        clients.append(client)
    return clients

def main():
    # --- Configuration ---
    CONFIG = {
        "NUM_CLIENTS": 10,
        "NUM_MALICIOUS": 2,
        "NUM_ROUNDS": 20,
        "LOCAL_EPOCHS": 1,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.01,
        "TARGET_CLASS": 7,
        "POISONING_RATE": 0.5,
        "TRIGGER_EPOCHS": 50,
        "TRIGGER_LR": 0.01,
        "SEED": 42,
        "DATA_ROOT": "data",
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    # --- End Configuration ---

    set_seed(CONFIG['SEED'])
    print(f"Running A3FL experiment on MNIST on device: {CONFIG['DEVICE']}")

    # --- Dataset and Model ---
    dataset_adapter = MNISTAdapter(root=CONFIG['DATA_ROOT'], download=True)
    model_cls = lambda: LeNet5(num_classes=10)
    
    client_loaders = dataset_adapter.get_client_loaders(
        num_clients=CONFIG['NUM_CLIENTS'], strategy="iid", batch_size=CONFIG['BATCH_SIZE'], seed=CONFIG['SEED']
    )
    test_loader = dataset_adapter.get_test_loader(batch_size=CONFIG['BATCH_SIZE'])


    a3fl_trigger = A3FLTrigger(position=(22, 22), size=(5, 5), alpha=1.0, in_channels=1)

    selector = RandomSelector(poisoning_rate=CONFIG['POISONING_RATE'])

    # --- Server and Client Initialization ---
    server = FedAvgAggregator(model=model_cls().to(CONFIG['DEVICE']), testloader=test_loader, device=CONFIG['DEVICE'])
    clients = build_clients(client_loaders, test_loader, model_cls, CONFIG, selector, a3fl_trigger)

    # --- Main FL Loop ---
    for round_idx in range(CONFIG['NUM_ROUNDS']):
        print(f"\n--- Round {round_idx + 1}/{CONFIG['NUM_ROUNDS']} ---")
        
        model_params = server.get_params()
        
        for client in clients:
            client.set_params(model_params)
            update = client.local_train(epochs=CONFIG['LOCAL_EPOCHS'], round_idx=round_idx)
            server.receive_update(update['weights'], length=update['num_samples'])
        
        server.aggregate()
        
        # --- Evaluation ---
        main_metrics = server.evaluate()
        print(f"Round {round_idx + 1}: Main Accuracy = {main_metrics['metrics']['main_accuracy']:.4f}")
        
        asr = evaluate_asr(server.model, test_loader.dataset, a3fl_trigger, CONFIG['TARGET_CLASS'], CONFIG['DEVICE'])
        print(f"Round {round_idx + 1}: Backdoor Accuracy = {asr:.4f}")

    print("\n--- Training Finished ---")
    final_main_metrics = server.evaluate()
    print(f"Final Main Accuracy: {final_main_metrics['metrics']['main_accuracy']:.4f}")
    final_asr = evaluate_asr(server.model, test_loader.dataset, a3fl_trigger, CONFIG['TARGET_CLASS'], CONFIG['DEVICE'])
    print(f"Final Backdoor Accuracy (ASR): {final_asr:.4f}")

if __name__ == "__main__":
    main()

