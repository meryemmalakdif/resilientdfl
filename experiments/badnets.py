#!/usr/bin/env python3
"""
Runs a dedicated FedAvg simulation for the BadNets backdoor attack on FEMNIST.
"""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import framework components
from src.fl.baseclient import BenignClient
from src.fl.baseserver import FedAvgAggregator
from src.datasets.femnist import FEMNISTAdapter
from src.models.lenet import LeNet5

# Import BadNets specific components
from src.attacks.badnets_client import BadNetsClient
from src.attacks.triggers.patch import PatchTrigger
from src.attacks.selectors.randomselector import RandomSelector

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_clients(client_loaders, test_loader, model_cls, config, selector, trigger):
    """Builds a mix of benign and BadNets clients."""
    clients = []
    for cid, loader in client_loaders.items():
        if cid < config['NUM_MALICIOUS']:
            # Create a BadNets client
            client = BadNetsClient(
                id=cid,
                trainloader=loader,
                testloader=test_loader,
                model=model_cls(num_classes=62).to(config['DEVICE']),
                lr=config['LEARNING_RATE'],
                weight_decay=0.0,
                epochs=config['LOCAL_EPOCHS'],
                device=config['DEVICE'],
                selector=selector,
                trigger=trigger,
                target_class=config['TARGET_CLASS']
            )
        else:
            # Create a Benign client
            client = BenignClient(
                id=cid,
                trainloader=loader,
                testloader=test_loader,
                model=model_cls(num_classes=62).to(config['DEVICE']),
                lr=config['LEARNING_RATE'],
                weight_decay=0.0,
                epochs=config['LOCAL_EPOCHS'],
                device=config['DEVICE']
            )
        clients.append(client)
    return clients

def main():
    # --- Configuration ---
    CONFIG = {
        "NUM_CLIENTS": 10,
        "NUM_MALICIOUS": 5,      # Number of malicious clients
        "NUM_ROUNDS": 20,
        "LOCAL_EPOCHS": 1,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.05,
        "TARGET_CLASS": 5,       # The class to target with the backdoor
        "POISONING_RATE": 0.10,   # Poison 10% of a malicious client's data
        "SEED": 42,
        "DATA_ROOT": "data",
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    # --- End Configuration ---

    set_seed(CONFIG['SEED'])
    print(f"Running on device: {CONFIG['DEVICE']}, seed: {CONFIG['SEED']}")

    # Prepare dataset
    dataset_adapter = FEMNISTAdapter(root=CONFIG['DATA_ROOT'], train=True, download=True)
    client_loaders = dataset_adapter.get_client_loaders(
        num_clients=CONFIG['NUM_CLIENTS'], strategy="iid", batch_size=CONFIG['BATCH_SIZE'], seed=CONFIG['SEED']
    )
    test_loader = dataset_adapter.get_test_loader(batch_size=CONFIG['BATCH_SIZE'])

    # Setup attack components for BadNets
    badnets_selector = RandomSelector(poisoning_rate=CONFIG['POISONING_RATE'])
    # Using a bright patch for grayscale FEMNIST. `color` is a tuple.
    badnets_trigger = PatchTrigger(position=(25, 25), size=(3, 3), color=(2.0,))

    # Server setup
    global_model = LeNet5(num_classes=62).to(CONFIG['DEVICE'])
    server = FedAvgAggregator(model=global_model, testloader=test_loader, device=CONFIG['DEVICE'])

    # Client setup
    clients = build_clients(client_loaders, test_loader, LeNet5, CONFIG, badnets_selector, badnets_trigger)

    # Main federated training loop
    for round_idx in range(CONFIG['NUM_ROUNDS']):
        print(f"\n--- Round {round_idx + 1}/{CONFIG['NUM_ROUNDS']} ---")
        
        model_params = server.get_params()
        
        # Client training
        for client in clients:
            client.set_params(model_params)
            update = client.local_train(epochs=CONFIG['LOCAL_EPOCHS'], round_idx=round_idx)
            server.receive_update(update['weights'], length=update['num_samples'])
        
        # Server aggregation and evaluation
        server.aggregate()
        
        main_task_metrics = server.evaluate()
        print(f"Server evaluation (Round {round_idx + 1}): Main Accuracy = {main_task_metrics['metrics']['accuracy']:.4f}")
        
        if CONFIG['NUM_MALICIOUS'] > 0:
            backdoor_metrics = server.evaluate_backdoor(trigger=badnets_trigger, target_class=CONFIG['TARGET_CLASS'])
            print(f"Server evaluation (Round {round_idx + 1}): Backdoor Accuracy = {backdoor_metrics['metrics']['backdoor_accuracy']:.4f}")

    print("\n--- Training Finished ---")
    final_metrics = server.evaluate()
    print(f"Final server evaluation: Main Accuracy = {final_metrics['metrics']['accuracy']:.4f}")
    if CONFIG['NUM_MALICIOUS'] > 0:
        final_backdoor_metrics = server.evaluate_backdoor(trigger=badnets_trigger, target_class=CONFIG['TARGET_CLASS'])
        print(f"Final server evaluation: Backdoor Accuracy = {final_backdoor_metrics['metrics']['backdoor_accuracy']:.4f}")

if __name__ == "__main__":
    main()