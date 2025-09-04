from typing import Any, Dict
import torch
from torch.utils.data import DataLoader

from ..fl.baseclient import BenignClient
from ..datasets.backdoor import make_triggered_loader
from .selectors.base import BaseSelector
from .triggers.base import BaseTrigger

class BadNetsClient(BenignClient):
    """
    A malicious client specifically for the BadNets attack.

    This client uses a selector and a trigger, and implements the naive
    poisoning strategy by training on a dataloader created with the
    `make_triggered_loader` helper.
    """
    def __init__(
        self,
        
        # BadNets specific components
        selector: BaseSelector,
        trigger: BaseTrigger,
        target_class: int,
        # Pass all BenignClient arguments via kwargs
        **kwargs,
    ):
        # Initialize the parent BenignClient with all its required arguments
        super().__init__(**kwargs)

        # Store the attack components
        self.selector = selector
        self.trigger = trigger
        self.target_class = target_class

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Overrides the benign training process to perform a BadNets attack.
        """
        print(f"--- BadNets Client [{self.get_id()}] training for round {round_idx} ---")

        # Create a poisoned dataloader for training using the helper
        poisoned_dataloader = make_triggered_loader(
            base_dataset=self.trainloader.dataset,
            trigger=self.trigger,
            keep_label=False,
            forced_label=self.target_class,
            fraction=self.selector.poisoning_rate,
            batch_size=self.trainloader.batch_size,
            shuffle=True,
            num_workers=self.trainloader.num_workers
        )

        # Perform naive training on the poisoned dataloader
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

        # Return results in the format expected by the server
        metrics = self.local_evaluate()['metrics']
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }