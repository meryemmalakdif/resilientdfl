from typing import Any, Dict
import torch

from ..fl.baseclient import BenignClient
from ..datasets.backdoor import create_backdoor_train_loader
from .selectors.base import BaseSelector
from .triggers.distributed import DBATrigger

class DBAClient(BenignClient):
    """
    A malicious client for the Distributed Backdoor Attack (DBA).

    This client's logic is identical to a BadNets client; it trains naively
    on a poisoned dataset. The distinction is that it is initialized with a
    DBATrigger, which only applies a small part of a global trigger pattern.
    """
    def __init__(
        self,
        selector: BaseSelector,
        trigger: DBATrigger,
        target_class: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(trigger, DBATrigger):
            raise TypeError("DBAClient must be initialized with a DBATrigger instance.")
        self.selector = selector
        self.trigger = trigger
        self.target_class = target_class

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Overrides the benign training process to perform a DBA attack.
        """
        poisoned_dataloader = create_backdoor_train_loader(
            base_dataset=self.trainloader.dataset,
            selector=self.selector,
            trigger=self.trigger,
            target_class=self.target_class,
            batch_size=self.trainloader.batch_size,
            shuffle=True
        )
        
        # Perform standard training on the poisoned dataloader
        self.model.train()
        epoch_count = epochs if epochs is not None else self.epochs_default
        for _ in range(epoch_count):
            for inputs, targets in poisoned_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
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
