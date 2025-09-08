#!/usr/bin/env python3
"""
run_experiment.py

A generic, YAML-configurable script for running FL experiments.
Example usage:
PYTHONPATH=. python experiments/run_experiment.py --config experiments/configs/base_config.yaml
"""
import random
import copy
import argparse
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

# --- Import Framework Components ---
from src.fl.baseclient import BenignClient
from src.fl.baseserver import FedAvgAggregator
from src.datasets.mnist import MNISTAdapter
from src.datasets.femnist import FEMNISTAdapter
from src.models.lenet import LeNet5
from src.models.simple_cifar import SimpleCNN_CIFAR
from src.models.gtsrb_cnn import GTSRB_CNN
from src.datasets.backdoor import create_asr_test_loader
from src.utils import ResultsLogger

# --- Import Attacks ---
from src.attacks.badnets_client import BadNetsClient
from src.attacks.scaling_client import ScalingAttackClient
from src.attacks.neurotoxin_client import NeurotoxinClient
from src.attacks.a3fl_client import A3FLClient
from src.attacks.dba_client import DBAClient
from src.attacks.triggers.patch import PatchTrigger
from src.attacks.triggers.a3fl import A3FLTrigger
from src.attacks.triggers.distributed import DBATrigger
from src.attacks.selectors.randomselector import RandomSelector

# --- Import Defenses ---
from src.defenses.clip_dp import NormClippingServer, WeakDPServer
from src.defenses.deepsight import DeepSightServer
from src.defenses.flame import FlameServer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_asr(model, test_dataset, trigger, target_class, device, batch_size):
    if trigger is None: return 0.0
    backdoor_loader = create_asr_test_loader(test_dataset, trigger, target_class, batch_size)
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
    clients = []
    for cid, loader in client_loaders.items():
        client_kwargs = {
            'id': cid, 'trainloader': loader, 'testloader': test_loader,
            'model': model_cls().to(config['device']), 'lr': config['lr'],
            'weight_decay': 0.0, 'epochs': config['local_epochs'], 'device': config['device']
        }
        
        is_malicious = (cid < config['num_malicious'])
        
        if is_malicious:
            attack_type = config['attack']
            if attack_type == 'badnets':
                client = BadNetsClient(**client_kwargs, selector=selector, trigger=trigger, target_class=config['target_class'])
            elif attack_type == 'scaling':
                client = ScalingAttackClient(**client_kwargs, selector=selector, trigger=trigger, target_class=config['target_class'],
                                             attack_start_round=config['attack_start_round'], attack_end_round=config['attack_end_round'],
                                             scale_factor=config['scale_factor'], num_clients=config['num_clients'],
                                             num_malicious=config['num_malicious'])
            elif attack_type == 'neurotoxin':
                client = NeurotoxinClient(**client_kwargs, selector=selector, trigger=trigger, target_class=config['target_class'],
                                          attack_start_round=config['attack_start_round'], attack_end_round=config['attack_end_round'],
                                          mask_k_percent=config['mask_k_percent'], malicious_epochs=config['malicious_epochs'])
            elif attack_type == 'a3fl':
                client = A3FLClient(**client_kwargs, selector=selector, trigger=trigger, target_class=config['target_class'])
            elif attack_type == 'dba':
                # Each DBA client gets its own unique trigger shard
                client = DBAClient(**client_kwargs, selector=selector, trigger=trigger[cid], target_class=config['target_class'])
            else:
                 client = BenignClient(**client_kwargs)
        else:
            client = BenignClient(**client_kwargs)
        clients.append(client)
    return clients

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    
    logger = ResultsLogger(config['experiment_name'], config)
    
    print("--- Configuration ---")
    for key, val in config.items():
        print(f"{key}: {val}")
    print("---------------------")

    # --- Dataset and Model ---
    if config['dataset'] == 'mnist':
        adapter = MNISTAdapter(root="data", download=True)
        model_cls = lambda: LeNet5(num_classes=10)
        in_channels = 1
    elif config['dataset'] == 'femnist':
        adapter = FEMNISTAdapter(root="data", train=True, download=True)
        model_cls = lambda: LeNet5(num_classes=62)
        in_channels = 1
    else:
        raise NotImplementedError(f"Dataset '{config['dataset']}' not implemented.")

    client_loaders = adapter.get_client_loaders(config['num_clients'], "iid", config['batch_size'], config['seed'])
    test_loader = adapter.get_test_loader(config['batch_size'])

    # --- Trigger ---
    trigger = None
    if config['attack'] != 'none':
        if config['attack'] == 'a3fl':
            trigger = A3FLTrigger(size=(3, 3), in_channels=in_channels, 
                                  trigger_epochs=config['trigger_epochs'], trigger_lr=config['trigger_lr'])
        elif config['attack'] == 'dba':
            shard_locations = [(0, 0), (2, 0), (0, 2), (2, 2)] # 2x2 grid
            trigger = [DBATrigger(client_id=i, shard_locations=shard_locations, global_position=(24, 24),
                                  patch_size=(2, 2), color=(2.0,) if in_channels==1 else (1.,1.,0.))
                       for i in range(config['num_malicious'])]
        else:
            trigger = PatchTrigger(position=(25, 25), size=(3, 3), color=(2.0,) if in_channels==1 else (1.,1.,0.))

    # --- Server (with defense) ---
    server_model = model_cls().to(device)
    defense_type = config.get('defense', 'none')
    if defense_type == 'none':
        server = FedAvgAggregator(server_model, test_loader, device)
    elif defense_type == 'norm_clipping':
        server = NormClippingServer(server_model, test_loader, device, config)
    elif defense_type == 'weak_dp':
        server = WeakDPServer(server_model, test_loader, device, config)
    elif defense_type == 'deepsight':
        server = DeepSightServer(server_model, test_loader, device, config)
    elif defense_type == 'flame':
        server = FlameServer(server_model, test_loader, device, config)
    else:
        raise ValueError(f"Unknown defense: {defense_type}")

    # --- Clients ---
    selector = RandomSelector(config['poisoning_rate'])
    clients = build_clients(client_loaders, test_loader, model_cls, config, selector, trigger)

    # --- FL Loop ---
    prev_model_params = None
    for round_idx in range(config['num_rounds']):
        current_round = round_idx + 1
        print(f"\n--- Round {current_round}/{config['num_rounds']} ---")
        model_params = server.get_params()
        
        agg_grad = None
        if prev_model_params:
            agg_grad = {name: model_params[name].to(device) - prev_model_params[name].to(device) for name in model_params}
        prev_model_params = copy.deepcopy(model_params)

        is_attack_active_this_round = (config['attack'] != 'none' and 
                                       config['num_malicious'] > 0 and
                                       config['attack_start_round'] <= current_round <= config['attack_end_round'])
        
        for client in clients:
            client.set_params(model_params)
            extra_args = {'prev_global_grad': agg_grad} if isinstance(client, (NeurotoxinClient, ScalingAttackClient)) else {}
            update = client.local_train(config['local_epochs'], current_round, **extra_args)
            server.receive_update(update['weights'], update['num_samples'])
        
        server.aggregate()
        
        main_metrics = server.evaluate()
        main_acc = main_metrics['metrics']['main_accuracy']
        
        # For DBA, the eval_trigger must be the full, assembled pattern
        eval_trigger = PatchTrigger(position=(24, 24), size=(4, 4), color=(2.0,) if in_channels==1 else (1.,1.,0.)) if config['attack'] == 'dba' else trigger
        asr = evaluate_asr(server.model, test_loader.dataset, eval_trigger, config['target_class'], device, config['batch_size'])
        
        print(f"Round {current_round}: Main Acc = {main_acc:.4f}, Backdoor ASR = {asr:.4f}, Attack Active: {is_attack_active_this_round}")
        logger.log_round(current_round, main_acc, asr, is_attack_active_this_round)

    print("\n--- Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)

