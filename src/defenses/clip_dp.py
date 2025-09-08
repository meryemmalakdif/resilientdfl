import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import copy

from ..fl.baseserver import FedAvgAggregator

class NormClippingServer(FedAvgAggregator):
    """
    Implements a robust aggregation server that clips the L2 norm of client updates.

    This defense limits the maximum influence any single client can have on the
    global model in a given round, mitigating attacks like model scaling.
    """
    def __init__(self, model: nn.Module, testloader: DataLoader = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)
        
        self.config = config if config is not None else {}
        self.clipping_norm = self.config.get('clipping_norm', 5.0)
        self.eta = self.config.get('eta', 1.0) # Server-side learning rate
        
        # Keep a CPU copy of the global model params for calculating diffs
        self.global_model_params = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        print(f"Initialized NormClippingServer with clipping_norm={self.clipping_norm}, eta={self.eta}")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Clips client updates before performing federated averaging.
        """
        if not self.updates:
            print("Warning: No updates to aggregate.")
            return self.get_params()

        # --- Step 1: Calculate the difference (update) for each client ---
        client_diffs = []
        client_num_samples = []
        for update in self.updates:
            diff = {name: param.to(self.device) - self.global_model_params[name].to(self.device)
                    for name, param in update['weights'].items() if not name.endswith('num_batches_tracked')}
            client_diffs.append(diff)
            client_num_samples.append(update['length'])

        # --- Step 2: Clip each client's update based on its L2 norm ---
        for diff in client_diffs:
            # Flatten all tensors to calculate the total norm of the update
            flat_diff = torch.cat([p.flatten() for p in diff.values()])
            diff_norm = torch.linalg.norm(flat_diff)

            if diff_norm > self.clipping_norm:
                scaling_factor = self.clipping_norm / diff_norm
                for name in diff:
                    diff[name].mul_(scaling_factor)
        
        # --- Step 3: Perform standard federated averaging on the clipped diffs ---
        total_samples = sum(client_num_samples)
        weight_accumulator = {name: torch.zeros_like(param) for name, param in self.model.state_dict().items()}

        for i, diff in enumerate(client_diffs):
            weight_fraction = client_num_samples[i] / total_samples
            for name, param_diff in diff.items():
                if name in weight_accumulator:
                    weight_accumulator[name].add_(param_diff * weight_fraction)

        # --- Step 4: Apply the aggregated update to the global model ---
        final_state_dict = self.model.state_dict()
        for name, param in final_state_dict.items():
            if name in weight_accumulator:
                param.add_(weight_accumulator[name] * self.eta)
        
        self.model.load_state_dict(final_state_dict)
        self.updates.clear()
        self.global_model_params = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        return self.get_params()


class WeakDPServer(NormClippingServer):
    """
    Implements a server that adds Gaussian noise for differential privacy.

    This defense builds upon Norm Clipping by adding noise to the final aggregated
    global model, which helps to obscure the contributions of any single client.
    """
    def __init__(self, model: nn.Module, testloader: DataLoader = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device, config)
        
        self.std_dev = self.config.get('dp_std_dev', 0.025)
        if self.std_dev < 0:
            raise ValueError("Standard deviation for DP noise must be non-negative.")
        print(f"Initialized WeakDPServer with std_dev={self.std_dev}")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        First performs clipping and aggregation, then adds Gaussian noise.
        """
        # Perform the norm clipping and aggregation from the parent class
        super().aggregate()

        # Add Gaussian noise to the updated global model parameters
        with torch.no_grad():
            final_state_dict = self.model.state_dict()
            for name, param in final_state_dict.items():
                if 'weight' in name or 'bias' in name:
                    noise = torch.normal(0, self.std_dev, param.shape, device=param.device)
                    param.add_(noise)
            self.model.load_state_dict(final_state_dict)

        # Update the stored params again after adding noise
        self.global_model_params = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        return self.get_params()
