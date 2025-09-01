from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence
import torch
import copy

class ModelPoisoner(ABC):
    @abstractmethod
    def poison_and_train(self, client, selector, trigger, **kwargs) -> Dict[str, Any]:
        """
        Orchestrate poisoning on `client` and return update dict similar to benign client:
        {
            'client_id': id,
            'weights': state_dict,
            'num_samples': num_samples,
            'metrics': {...}
        }
        """
        pass

class NaiveDataPoisoner(ModelPoisoner):
    """Replace selected samples with triggered inputs + target label, then do normal local training."""
    def __init__(self, target_label:int, poison_fraction:float=0.1):
        self.target_label = int(target_label)
        self.poison_fraction = poison_fraction

    def poison_and_train(self, client, selector, trigger, **kwargs):
        # 1) decide indices to poison
        local_indices = list(range(len(client.trainloader.dataset)))
        num_poison = max(1, int(len(local_indices) * self.poison_fraction))
        chosen = selector.select(local_indices, labels=getattr(client.trainloader.dataset, "targets", None), num_poison=num_poison)

        # 2) create poisoned dataset wrapper (non-destructive) that applies trigger and label change on-the-fly
        from torch.utils.data import Dataset
        class PoisonedWrapper(Dataset):
            def __init__(self, base, chosen_set, trigger, target_label):
                self.base = base
                self.chosen = set(chosen_set)
                self.trigger = trigger
                self.target = target_label
            def __len__(self):
                return len(self.base)
            def __getitem__(self, idx):
                x, y = self.base[idx]
                if idx in self.chosen:
                    x = self.trigger.apply(x)
                    y = self.target
                return x, y

        poisoned_ds = PoisonedWrapper(client.trainloader.dataset, chosen, trigger, self.target_label)
        poisoned_loader = torch.utils.data.DataLoader(poisoned_ds, batch_size=client.trainloader.batch_size, shuffle=True, num_workers=client.trainloader.num_workers)

        # 3) run local training on poisoned_loader (use client's model/optimizer but DO NOT overwrite client.trainloader)
        # we'll temporarily swap
        orig_loader = client.trainloader
        client.trainloader = poisoned_loader
        update = client.local_train(epochs=kwargs.get("epochs", 1), round_idx=kwargs.get("round_idx", 0))
        client.trainloader = orig_loader
        return update

class ModelReplacementPoisoner(ModelPoisoner):
    """
    Train locally on poisoned data, then perform model replacement:
    new_global = global + gamma * (local_weights - global)
    gamma can be used to fully replace server model.
    """
    def __init__(self, gamma: float = 1.0, target_label:int = 0, poison_fraction:float = 0.1):
        self.gamma = gamma
        self.target_label = target_label
        self.poison_fraction = poison_fraction

    def poison_and_train(self, client, selector, trigger, **kwargs):
        # similar poisoned data setup as above
        upd = NaiveDataPoisoner(self.target_label, self.poison_fraction).poison_and_train(client, selector, trigger, **kwargs)
        # perform model replacement scaling
        global_params = kwargs.get("global_params")
        if global_params is None:
            return upd
        local_weights = upd["weights"]
        averaged = {}
        for k in local_weights:
            averaged[k] = global_params[k].cpu().float() + self.gamma * (local_weights[k].cpu().float() - global_params[k].cpu().float())
        upd["weights"] = {k: v.clone() for k, v in averaged.items()}
        return upd
