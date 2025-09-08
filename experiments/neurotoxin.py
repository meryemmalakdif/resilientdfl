#!/usr/bin/env python3
import random
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Import framework components
from src.fl.baseclient import BenignClient
from src.fl.baseserver import FedAvgAggregator
from src.datasets.femnist import FEMNISTAdapter
from src.models.lenet import LeNet5
from src.datasets.backdoor import create_asr_test_loader

# Import Neurotoxin specific components
from src.attacks.neurotoxin_client import NeurotoxinClient
from src.attacks.triggers.patch import PatchTrigger
from src.attacks.selectors.randomselector import RandomSelector

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def model_to_vec_numpy(model: torch.nn.Module) -> np.ndarray:
    """
    Flatten model parameters into a single numpy 1D array on CPU (float32).
    Order matches model.parameters() iteration.
    """
    parts = []
    for p in model.parameters():
        parts.append(p.detach().cpu().float().view(-1))
    if len(parts) == 0:
        return np.array([], dtype=np.float32)
    return torch.cat(parts).numpy()

def evaluate_asr(model: torch.nn.Module, test_dataset: Dataset, trigger, target_label: int, device: torch.device, batch_size: int = 256):
    """
    Evaluates the Attack Success Rate (ASR) of the model on a triggered test set.
    """
    # Use the helper to create a fully poisoned test loader
    backdoor_loader = create_asr_test_loader(
        base_dataset=test_dataset,
        trigger=trigger,
        target_class=target_label,
        batch_size=batch_size)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in backdoor_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    asr = (correct / total) if total > 0 else 0.0
    return asr

def build_clients(client_loaders, test_loader, model_cls, config, selector, trigger):
    """Builds a mix of benign and Neurotoxin clients."""
    clients = []
    for cid, loader in client_loaders.items():
        client_kwargs = {
            'id': cid,
            'trainloader': loader,
            'testloader': test_loader,
            'model': model_cls(num_classes=62).to(config['DEVICE']),
            'lr': config['LEARNING_RATE'],
            'weight_decay': 0.0,
            'epochs': config['LOCAL_EPOCHS'],
            'device': config['DEVICE']
        }
        if cid < config['NUM_MALICIOUS']:
            client = NeurotoxinClient(
                **client_kwargs,
                selector=selector,
                trigger=trigger,
                target_class=config['TARGET_CLASS'],
                attack_start_round=config['ATTACK_START_ROUND'],
                topk_ratio=config['MASK_K_PERCENT'],            # using Neurotoxin naming
                num_sample_for_mask=config.get('NUM_SAMPLE_MASK', 64),
                norm_threshold=config.get('NORM_THRESHOLD', 0.2)
            )
        else:
            client = BenignClient(**client_kwargs)
        clients.append(client)
    return clients

def main():
    CONFIG = {
        "NUM_CLIENTS": 10,
        "NUM_MALICIOUS": 4,
        "NUM_ROUNDS": 5,
        "ATTACK_START_ROUND": 0, # Attack starts on round 0
        "MASK_K_PERCENT": 0.05,  # topk_ratio (fraction of coords to KEEP)
        "LOCAL_EPOCHS": 1,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.05,
        "TARGET_CLASS": 5,
        "POISONING_RATE": 0.3,
        "SEED": 42,
        "DATA_ROOT": "data",
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    set_seed(CONFIG['SEED'])
    print(f"Running on device: {CONFIG['DEVICE']}, seed: {CONFIG['SEED']}")

    # Prepare dataset
    dataset_adapter = FEMNISTAdapter(root=CONFIG['DATA_ROOT'], train=True, download=True)
    client_loaders = dataset_adapter.get_client_loaders(
        num_clients=CONFIG['NUM_CLIENTS'], strategy="iid", batch_size=CONFIG['BATCH_SIZE'], seed=CONFIG['SEED']
    )
    test_loader = dataset_adapter.get_test_loader(batch_size=CONFIG['BATCH_SIZE'])

    # Setup attack components
    attack_selector = RandomSelector(poisoning_rate=CONFIG['POISONING_RATE'])
    # Use normalized patch color (1.0) for float tensors in [0,1]; change if your transforms differ
    attack_trigger = PatchTrigger(position=(25, 25), size=(3, 3), color=(1.0,))

    # Server setup
    global_model = LeNet5(num_classes=62).to(CONFIG['DEVICE'])
    server = FedAvgAggregator(model=global_model, testloader=test_loader, device=CONFIG['DEVICE'])

    # Client setup
    clients = build_clients(client_loaders, test_loader, LeNet5, CONFIG, attack_selector, attack_trigger)
    
    # Main federated training loop
    for round_idx in range(CONFIG['NUM_ROUNDS']):
        print(f"\n--- Round {round_idx + 1}/{CONFIG['NUM_ROUNDS']} ---")
        
        # get the current global params (server.model already holds the up-to-date global model)
        # vectorize global model (CPU numpy vector) to pass to malicious clients for clipping
        global_state_vec = model_to_vec_numpy(server.model)

        # Get state_dict to broadcast (clients expect state_dict in set_params)
        model_params = server.get_params()

        for client in clients:
            client.set_params(model_params)  # ensure client model == global
            if isinstance(client, NeurotoxinClient):
                # pass the global vector for clipping/reference
                update = client.local_train(
                    epochs=CONFIG['LOCAL_EPOCHS'], 
                    round_idx=round_idx, 
                    global_state_vec=global_state_vec
                )
            else:
                update = client.local_train(
                    epochs=CONFIG['LOCAL_EPOCHS'], 
                    round_idx=round_idx
                )

            server.receive_update(update['weights'], length=update['num_samples'])
        
        # aggregate selected updates and update server.model internally
        server.aggregate()
        
        main_metrics = server.evaluate()

        try:
            print(f"Round {round_idx + 1}: Main Accuracy = {main_metrics['metrics']['main_accuracy']:.4f}")
        except Exception:
            print(f"Round {round_idx + 1}: Main metrics = {main_metrics}")

        # Evaluate ASR on fully triggered test set
        asr = evaluate_asr(server.model, test_loader.dataset, attack_trigger, CONFIG['TARGET_CLASS'], CONFIG['DEVICE'])
        print(f"Round {round_idx + 1}: Backdoor Accuracy = {asr:.4f}")

    print("\n--- Training Finished ---")
    final_main_metrics = server.evaluate()
    final_asr = evaluate_asr(server.model, test_loader.dataset, attack_trigger, CONFIG['TARGET_CLASS'], CONFIG['DEVICE'])
    try:
        print(f"Final Main Accuracy: {final_main_metrics['metrics']['main_accuracy']:.4f}")
    except Exception:
        print(f"Final Main Metrics: {final_main_metrics}")
    print(f"Final Backdoor Accuracy (ASR): {final_asr:.4f}")


if __name__ == "__main__":
    main()
