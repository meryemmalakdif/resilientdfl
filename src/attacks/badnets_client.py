from typing import Any, Dict
import torch
from torch.utils.data import DataLoader

from ..fl.baseclient import BenignClient
from ..datasets.backdoor import create_backdoor_train_loader
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

        poisoned_dataloader = create_backdoor_train_loader(
            base_dataset=self.trainloader.dataset,
            selector=self.selector,
            trigger=self.trigger,
            target_class=self.target_class,
            batch_size=self.trainloader.batch_size,
            shuffle=True)
        self.trainloader = poisoned_dataloader