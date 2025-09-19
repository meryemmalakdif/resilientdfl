import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
import copy

# --- Import Framework Components ---
from src.fl.baseclient import BenignClient
from src.fl.baseserver import FedAvgAggregator
from src.datasets.GTSRB import GTSRBAdapter
from src.models.gtsrb_cnn import GTSRB_CNN
from src.datasets.backdoor import create_asr_test_loader

# --- Import IBA Components ---
from src.attacks.iba_client import IBAClient
from src.attacks.triggers.iba import IBATrigger
from src.models.unet import UNet
from src.attacks.selectors.randomselector import RandomSelector

# --- MODIFICATION START: Worker now returns the updated trigger ---
def client_training_worker(args_tuple):
    """
    Wrapper function for parallel execution. It now also returns the
    client's trigger object after training.
    """
    client, model_params, epochs, current_round = args_tuple
    
    client.set_params(copy.deepcopy(model_params))
    update = client.local_train(epochs=epochs, round_idx=current_round)
    
    # Return the client object itself to access its updated trigger in the main process
    return (update['weights'], update['num_samples'], client)
# --- MODIFICATION END ---


def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_asr(model: torch.nn.Module, test_dataset: Dataset, trigger: IBATrigger, target_label: int, device: torch.device, batch_size: int = 256):
    """
    Evaluates the Attack Success Rate (ASR) of the model on a triggered test set.
    """
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
    
    asr = (correct / total) if total > 0 else 0.0
    return asr

def build_clients(client_loaders, test_loader, model_cls, config, selector, trigger):
    """Builds a mix of benign and IBA clients."""
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
            client = IBAClient(
                **client_kwargs,
                selector=selector,
                trigger=trigger,
                target_class=config['TARGET_CLASS'],
                attack_start_round=config['ATTACK_START_ROUND'],
                attack_end_round=config['ATTACK_END_ROUND']
            )
        else:
            client = BenignClient(**client_kwargs)
        clients.append(client)
    return clients

def main():
    # --- Configuration ---
    CONFIG = {
        "NUM_PARALLEL_CLIENTS": 8,
        "NUM_CLIENTS": 10,
        "NUM_MALICIOUS": 2,
        "NUM_ROUNDS": 20,
        "LOCAL_EPOCHS": 1,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.01,
        "TARGET_CLASS": 7,
        "POISONING_RATE": 0.20,
        "ATTACK_START_ROUND": 1,
        "ATTACK_END_ROUND": 20,
        "SEED": 42,
        "DATA_ROOT": "data",
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    # --- End Configuration ---

    set_seed(CONFIG['SEED'])
    print(f"Running IBA experiment on GTSRB on device: {CONFIG['DEVICE']}")

    dataset_adapter = GTSRBAdapter(root=CONFIG['DATA_ROOT'])
    model_cls = lambda: GTSRB_CNN(num_classes=43)
    
    # NOTE: Your GTSRBAdapter needs to accept `num_workers`
    client_loaders = dataset_adapter.get_client_loaders(
        num_clients=CONFIG['NUM_CLIENTS'], strategy="iid", batch_size=CONFIG['BATCH_SIZE'], seed=CONFIG['SEED']
    )
    test_loader = dataset_adapter.get_test_loader(batch_size=CONFIG['BATCH_SIZE'])

    unet_generator = UNet(out_channel=3)
    iba_trigger = IBATrigger(unet_model=unet_generator)
    selector = RandomSelector(poisoning_rate=CONFIG['POISONING_RATE'])

    server = FedAvgAggregator(model=model_cls().to(CONFIG['DEVICE']), testloader=test_loader, device=CONFIG['DEVICE'])
    clients = build_clients(client_loaders, test_loader, model_cls, CONFIG, selector, iba_trigger)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with mp.Pool(processes=CONFIG['NUM_PARALLEL_CLIENTS']) as pool:
        for round_idx in range(CONFIG['NUM_ROUNDS']):
            current_round = round_idx + 1
            print(f"\n--- Round {current_round}/{CONFIG['NUM_ROUNDS']} ---")
            
            model_params = server.get_params()
            
            worker_args = [
                (client, model_params, CONFIG['LOCAL_EPOCHS'], current_round)
                for client in clients
            ]

            print(f"Starting parallel training for {len(clients)} clients...")
            results = pool.map(client_training_worker, worker_args)
            
            # --- MODIFICATION START: Collect results and capture the updated trigger ---
            # We'll use the trigger from the first malicious client for evaluation.
            trigger_for_eval = None 
            
            for weights, num_samples, trained_client in results:
                server.receive_update(weights, num_samples)
                # If this client is malicious, its trigger has been updated.
                # We save the first one we find to use for this round's ASR evaluation.
                if isinstance(trained_client, IBAClient) and trigger_for_eval is None:
                    trigger_for_eval = trained_client.trigger
            
            # If for some reason no malicious client was in the pool, fallback to original.
            if trigger_for_eval is None:
                trigger_for_eval = iba_trigger
            # --- MODIFICATION END ---
            
            server.aggregate()
            
            main_metrics = server.evaluate()
            print(f"Round {current_round}: Main Accuracy = {main_metrics['metrics']['main_accuracy']:.4f}")
            
            # --- MODIFICATION: Use the trigger that was updated in this round for evaluation ---
            asr = evaluate_asr(server.model, test_loader.dataset, trigger_for_eval, CONFIG['TARGET_CLASS'], CONFIG['DEVICE'])
            print(f"Round {current_round}: Backdoor Accuracy = {asr:.4f}")

    print("\n--- Training Finished ---")
    final_main_metrics = server.evaluate()
    final_asr = evaluate_asr(server.model, test_loader.dataset, trigger_for_eval, CONFIG['TARGET_CLASS'], CONFIG['DEVICE'])
    print(f"Final Main Accuracy: {final_main_metrics['metrics']['main_accuracy']:.4f}")
    print(f"Final Backdoor Accuracy (ASR): {final_asr:.4f}")

if __name__ == "__main__":
    main()

