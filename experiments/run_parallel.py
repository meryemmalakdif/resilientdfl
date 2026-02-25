#!/usr/bin/env python3
"""
run_experiment.py

A generic, YAML-configurable script for running FL experiments with parallel client training.
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
import multiprocessing as mp
import time

# --- Import Framework Components ---
from src.fl.baseclient import BenignClient
from src.fl.baseserver import FedAvgAggregator
from src.datasets.mnist import MNISTAdapter
from src.datasets.femnist import FEMNISTAdapter
from src.datasets.cifar10 import CIFAR10Adapter
from src.datasets.GTSRB import GTSRBAdapter
from src.models.lenet import LeNet5, EMNIST_CNN
from src.models.simple import CifarNetGN
from src.models.gtsrb_cnn import GTSRB_CNN
from src.datasets.backdoor import create_asr_test_loader
from src.utils import ResultsLogger

# --- Import Attacks ---
from src.attacks.badnets_client import BadNetsClient
from src.attacks.scaling_client import ScalingAttackClient
from src.attacks.neurotoxin_client import NeurotoxinClient
from src.attacks.a3fl_client import A3FLClient
from src.attacks.dba_client import DBAClient
from src.attacks.iba_client import IBAClient
from src.attacks.triggers.patch import PatchTrigger
from src.attacks.triggers.a3fl import A3FLTrigger
from src.attacks.triggers.distributed import DBATrigger
from src.attacks.triggers.iba import IBATrigger
from src.models.unet import UNet, FEMNISTAutoencoder
from src.attacks.selectors.randomselector import RandomSelector

# --- Import Defenses ---
from src.defenses.clip_dp import NormClippingServer, WeakDPServer
from src.defenses.deepsight import DeepSightServer
from src.defenses.flame import FlameServer
from src.defenses.flame_new1 import FlameServer1
from src.defenses.flame_new import RobustFlameServer
from src.defenses.nested_flame import FlameNestedServer
from src.defenses.layer_wise_flame import LayerFlameServer
from src.defenses.krum import MKrumServer

import inspect, os, sys, traceback

# --- Worker function for multiprocessing ---
def client_training_worker(args_tuple):
    """
    Wrapper function for parallel execution. Returns serializable results,
    including the updated state of any stateful triggers.
    """
    client, model_params, config, current_round, extra_args = args_tuple
    client.set_params(copy.deepcopy(model_params))
    update = client.local_train(config['local_epochs'], current_round, **extra_args)
    
    # --- MODIFICATION: Safely extract trigger state for different trigger types ---
    updated_trigger_state = None
    is_attack_round = (config['attack_start_round'] <= current_round <= config['attack_end_round'])
    
    if is_attack_round:
        if isinstance(client, IBAClient):
            state_dict = client.trigger.generator.state_dict()
            updated_trigger_state = {k: v.cpu().clone() for k, v in state_dict.items()}
        elif isinstance(client, A3FLClient):
            updated_trigger_state = client.trigger.pattern.cpu().clone()
        
    return (update['weights'], update['num_samples'], updated_trigger_state)

# def client_training_worker(args_tuple):
#     client, model_params, config, current_round, extra_args = args_tuple
#     client.set_params(copy.deepcopy(model_params))

#     # DEBUG: log which worker will run what
#     try:
#         print(f"[worker pid={os.getpid()}] Calling local_train for client type={type(client).__name__} id={getattr(client, 'id', 'N/A')} round={current_round} extra_args={list(extra_args.keys())}", flush=True)
#     except Exception:
#         print(f"[worker pid={os.getpid()}] Calling local_train (failed to get client info)", flush=True)


#     # Build kwargs only for parameters that the client's local_train accepts
#     sig = inspect.signature(client.local_train)
#     allowed_kwargs = {}
#     for k, v in extra_args.items():
#         if k in sig.parameters:
#             allowed_kwargs[k] = v

#     # update = client.local_train(config['local_epochs'], current_round, **allowed_kwargs)

#     try:
#         update = client.local_train(config['local_epochs'], current_round, **allowed_kwargs)
#         print(f"[worker pid={os.getpid()}] Finished local_train for client id={getattr(client, 'id', 'N/A')}", flush=True)
#     except Exception as e:
#         # Print full traceback to stdout so you see errors from child processes
#         print(f"[worker pid={os.getpid()}] Exception in client.local_train: {e}", flush=True)
#         traceback.print_exc(file=sys.stdout)
#         # re-raise so Pool can notice the failure (optional)
#         raise

#     # --- trigger state extraction unchanged ---
#     updated_trigger_state = None
#     is_attack_round = (config['attack_start_round'] <= current_round <= config['attack_end_round'])

#     if is_attack_round:
#         if isinstance(client, IBAClient):
#             state_dict = client.trigger.generator.state_dict()
#             updated_trigger_state = {k: v.cpu().clone() for k, v in state_dict.items()}
#         elif isinstance(client, A3FLClient):
#             updated_trigger_state = client.trigger.pattern.cpu().clone()

#     return (update['weights'], update['num_samples'], updated_trigger_state)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# def evaluate_asr(model, test_dataset, trigger, target_class, device, batch_size):
#     if trigger is None: return 0.0
#     backdoor_loader = create_asr_test_loader(test_dataset, trigger, target_class, batch_size)
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for inputs, targets in backdoor_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs.data, 1)
#             correct += (preds == targets).sum().item()
#             total += targets.size(0)
#     return (correct / total) if total > 0 else 0.0

def evaluate_asr(model, backdoor_loader: DataLoader, device: torch.device):
    if backdoor_loader is None: return 0.0
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
    """
    A robust factory function to build a mix of benign and malicious clients.
    """
    clients = []
    for cid, loader in client_loaders.items():
        base_kwargs = {
            'id': cid, 'trainloader': loader, 'testloader': test_loader,
            'model': model_cls().to(config['device']), 'lr': config['lr'],
            'weight_decay': 0.0, 'epochs': config['local_epochs'], 'device': config['device']
        }
        
        if cid < config['num_malicious']:
            attack_type = config['attack']
            malicious_kwargs = {'selector': selector, 'target_class': config['target_class']}
            
            if attack_type == 'badnets':
                client = BadNetsClient(**base_kwargs, **malicious_kwargs, trigger=trigger)
            elif attack_type == 'scaling':
                client = ScalingAttackClient(**base_kwargs, **malicious_kwargs, trigger=trigger,
                                             attack_start_round=config['attack_start_round'], attack_end_round=config['attack_end_round'],
                                             scale_factor=config['scale_factor'], num_total_clients=config['num_clients'],
                                             num_malicious_clients=config['num_malicious'], num_malicious_epochs=config['malicious_epochs'])
            elif attack_type == 'neurotoxin':
                client = NeurotoxinClient(**base_kwargs, **malicious_kwargs, trigger=trigger,
                                          attack_start_round=config['attack_start_round'], attack_end_round=config['attack_end_round'],
                                          mask_k_percent=config.get('mask_k_percent', 0.9))
            elif attack_type == 'a3fl':
                client = A3FLClient(**base_kwargs, **malicious_kwargs, trigger=trigger,
                                    attack_start_round=config['attack_start_round'], attack_end_round=config['attack_end_round'])
            elif attack_type == 'iba':
                client = IBAClient(**base_kwargs, **malicious_kwargs, trigger=trigger,
                                   attack_start_round=config['attack_start_round'], attack_end_round=config['attack_end_round'])
            elif attack_type == 'dba':
                client = DBAClient(**base_kwargs, **malicious_kwargs, trigger=trigger[cid],
                                   attack_start_round=config['attack_start_round'], attack_end_round=config['attack_end_round'])
            else:
                 client = BenignClient(**base_kwargs)
        else:
            client = BenignClient(**base_kwargs)
        clients.append(client)
    return clients

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    

    torch.cuda.empty_cache()

    
    logger = ResultsLogger(config['experiment_name'], config)
    
    print("--- Configuration ---"); [print(f"{k}: {v}") for k, v in config.items()]; print("---------------------")

    # --- Dataset and Model ---
    if config['dataset'] == 'mnist':
        adapter = MNISTAdapter(root="data", download=True)
        model_cls = lambda: LeNet5(num_classes=10)
        in_channels = 1
        image_size = (28, 28)
    elif config['dataset'] == 'femnist':
        adapter = FEMNISTAdapter(root="data", train=True, download=True)
        model_cls = lambda: EMNIST_CNN(num_classes=62)
        in_channels = 1
        image_size = (28, 28)
    elif config['dataset'] == 'cifar10':
        adapter = CIFAR10Adapter(root="data", download=True)
        model_cls = lambda: CifarNetGN()
        in_channels = 3
        image_size = (32, 32)
    elif config['dataset'] == 'gtsrb':
        adapter = GTSRBAdapter(root="data")
        model_cls = lambda: GTSRB_CNN(num_classes=43)
        in_channels = 3
        image_size = (32, 32)
    else:
        raise NotImplementedError(f"Dataset '{config['dataset']}' not implemented.")

    client_loaders = adapter.get_client_loaders(config['num_clients'], "iid", config['batch_size'], config['seed'])
    test_loader = adapter.get_test_loader(config['test_batch_size'])

    # --- Trigger ---
    trigger = None
    if config['attack'] != 'none':
        trigger_pos = (image_size[0] - 4, image_size[1] - 4)
        if config['attack'] == 'a3fl':
            trigger = A3FLTrigger(position=trigger_pos, size=(3, 3), in_channels=in_channels, image_size=image_size,
                                  trigger_epochs=config.get('trigger_epochs', 5), trigger_lr=config.get('trigger_lr', 0.01))
        elif config['attack'] == 'iba':
            if config['dataset'] == 'femnist':
                unet_generator = FEMNISTAutoencoder(in_channel=1, out_channel=1)
            else:
                unet_generator = UNet(in_channel=in_channels, out_channel=in_channels)
            trigger = IBATrigger(unet_model=unet_generator)
        elif config['attack'] == 'dba':
            shard_locations = [(0, 0), (2, 0), (0, 2), (2, 2)]
            trigger = [DBATrigger(client_id=i, shard_locations=shard_locations, global_position=(image_size[0]-5, image_size[1]-5),
                                  patch_size=(2, 2), color=(1.0,)*in_channels)
                       for i in range(config['num_malicious'])]
        else:
            trigger = PatchTrigger(position=trigger_pos, size=(3, 3), color=(1.0,)*in_channels)

    # --- Server (with defense) ---
    server_model = model_cls().to(device)

    defense_type = config.get('defense', 'none')

    if defense_type == 'none':
        server = FedAvgAggregator(server_model, test_loader, device)
    # ... (other defenses) ...
    elif defense_type == 'krum':
        server = MKrumServer(server_model, test_loader, device, config)
    elif defense_type == 'clip':
        server = NormClippingServer(server_model, test_loader, device, config)
    elif defense_type == 'flame':
        server = FlameServer(model=server_model, testloader=test_loader, device=device, config= config)
    elif defense_type == 'flame1':
        server = FlameServer1(model=server_model, testloader=test_loader, device=device, config= config)    
    elif defense_type == 'flame_new':
        server = RobustFlameServer(model=server_model, testloader=test_loader, device=device, config= config)
    elif defense_type == 'flame_nested':
        server = FlameNestedServer(model=server_model, testloader=test_loader, device=device, config= config)
    elif defense_type == 'flame_layer':
        server = LayerFlameServer(model=server_model, testloader=test_loader, device=device, config= config)
      
   
    elif defense_type == 'deepsight':
        server = DeepSightServer(server_model, test_loader, device, config)
    else:
        raise ValueError(f"Unknown defense: {defense_type}")

    # --- Clients ---
    selector = RandomSelector(config['poisoning_rate'])
    clients = build_clients(client_loaders, test_loader, model_cls, config, selector, trigger)

    backdoor_loader_for_eval = None
    if config['attack'] != 'none':
        eval_trigger_for_dba = PatchTrigger(position=(image_size[0]-5, image_size[1]-5), size=(4, 4), color=(1.0,)*in_channels) if config['attack'] == 'dba' else None
        initial_eval_trigger = eval_trigger_for_dba if config['attack'] == 'dba' else trigger
        backdoor_loader_for_eval = create_asr_test_loader(test_loader.dataset, initial_eval_trigger, config['target_class'], config['batch_size'])


    # --- FL Loop with Parallelism ---
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    with mp.Pool(processes=config.get('num_parallel_clients', 1)) as pool:
        prev_model_params = None
        trigger_for_eval = trigger
        for round_idx in range(config['num_rounds']):
            round_start_time = time.time()
            current_round = round_idx + 1
            print(f"\n--- Round {current_round}/{config['num_rounds']} ---")
            
            model_params = server.get_params()
            
            agg_grad = None
            if prev_model_params:
                agg_grad = {name: model_params[name].to(device) - prev_model_params[name].to(device) for name in prev_model_params}
            prev_model_params = copy.deepcopy(model_params)

            is_attack_active_this_round = (config['attack'] != 'none' and 
                                           config['num_malicious'] > 0 and
                                           config['attack_start_round'] <= current_round <= config['attack_end_round'])
            
            worker_args = []
            for client in clients:
                extra_args = {'prev_global_grad': agg_grad} if isinstance(client, NeurotoxinClient) else {}
                worker_args.append((client, model_params, config, current_round, extra_args))

            results = pool.map(client_training_worker, worker_args)
            
            updated_trigger_state = None
            for weights, num_samples, trigger_state in results:
                server.receive_update(weights, num_samples)
                if trigger_state is not None and updated_trigger_state is None:
                    # Capture the first updated trigger state we receive from any worker
                    updated_trigger_state = trigger_state
            
            if updated_trigger_state is not None:
                # Based on the attack type, update the main trigger_for_eval object
                if config['attack'] == 'iba':
                    # If the state is a dictionary, it's a model state_dict
                    trigger_for_eval.generator.load_state_dict(updated_trigger_state)
                elif config['attack'] == 'a3fl':
                    # If the state is a tensor, it's the A3FL pattern
                    trigger_for_eval.pattern = updated_trigger_state
            
            is_stateful_attack = config['attack'] in ['iba', 'a3fl']
            if is_stateful_attack and is_attack_active_this_round and updated_trigger_state is not None:
                print("Stateful trigger updated. Recreating ASR test loader...")
                if config['attack'] == 'iba':
                    trigger_for_eval.generator.load_state_dict(updated_trigger_state)
                elif config['attack'] == 'a3fl':
                    trigger_for_eval.pattern = updated_trigger_state
                backdoor_loader_for_eval = create_asr_test_loader(test_loader.dataset, trigger_for_eval, config['target_class'], config['batch_size'])

            server.aggregate()
            
            main_metrics = server.evaluate()
            main_acc = main_metrics['metrics']['main_accuracy']
            
            if current_round >= config['attack_start_round']:
                asr = evaluate_asr(server.model, backdoor_loader_for_eval, device)
            else: 
                asr = 0.0000
            
            print(f"Round {current_round}: Main Acc = {main_acc:.4f}, Backdoor ASR = {asr:.4f}, Attack Active: {is_attack_active_this_round}, took {time.time() - round_start_time} seconds")
            logger.log_round(current_round, main_acc, asr, is_attack_active_this_round)

    print("\n--- Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)

