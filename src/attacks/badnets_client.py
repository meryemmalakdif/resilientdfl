from typing import Any, Dict
import torch
from torch.utils.data import DataLoader

from ..fl.baseclient import BenignClient
from ..datasets.backdoor import PoisonedDataset
from .selectors.base import BaseSelector
from .triggers.base import BaseTrigger

class BadNetsClient(BenignClient):
    def __init__(self, selector: BaseSelector, trigger: BaseTrigger, target_class: int, **kwargs):
        super().__init__(**kwargs)
        self.selector = selector
        self.trigger = trigger
        self.target_class = target_class

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        # print(f"\n--- BadNets Client [{self.get_id()}] starting training for round {round_idx} ---")
        
        poisoned_indices = self.selector.select(self.trainloader.dataset)
        poisoned_dataset = PoisonedDataset(
            self.trainloader.dataset, set(poisoned_indices), self.trigger, self.target_class
        )
        poisoned_dataloader = DataLoader(poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)

        self._model.train()
        epoch_count = epochs if epochs is not None else self.epochs_default
        for _ in range(epoch_count):
            for inputs, targets in poisoned_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        metrics = self.local_evaluate()['metrics']
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }
    

