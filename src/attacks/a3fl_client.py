from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

import copy as cp

from ..fl.baseclient import BenignClient
from ..datasets.backdoor import create_backdoor_train_loader
from .selectors.base import BaseSelector
from .triggers.a3fl import A3FLTrigger


class A3FLClient(BenignClient):
    """
    A3FL-style malicious client.

    Workflow:
      1) Use a small sampled subset of local clean data to train the A3FL trigger
         (the trigger's `train_trigger` method builds an adversarial model internally).
      2) Create a backdoor training loader mixing poisoned & clean data (using
         create_backdoor_train_loader helper) and run local training on it.
      3) Return the same dict structure as BenignClient.local_train, so server
         can receive and aggregate updates normally.
    """

    def __init__(
        self,
        selector: BaseSelector,
        trigger: A3FLTrigger,
        target_class: int,
        # optional tuning hyperparams for trigger training and sampling:
        trigger_epochs: int = 50,
        trigger_lr: float = 1e-2,
        trigger_lambda_balance: float = 0.1,
        trigger_sample_size: int = 256,
        # pass BenignClient args via kwargs
        **kwargs,
    ):
        # Initialize parent
        super().__init__(**kwargs)

        # Attack components
        if not isinstance(trigger, A3FLTrigger):
            raise ValueError("trigger must be an A3FLTrigger instance")
        self.selector = selector
        self.trigger: A3FLTrigger = trigger
        self.target_class = int(target_class)

        # A3FL-specific hyperparams (tunable)
        self.trigger_epochs = int(trigger_epochs)
        self.trigger_lr = float(trigger_lr)
        self.trigger_lambda_balance = float(trigger_lambda_balance)
        self.trigger_sample_size = int(trigger_sample_size)

        # internal placeholder for the poisoned loader used during training
        self._poisoned_loader: Optional[DataLoader] = None

    def _build_trigger_dataloader(self) -> DataLoader:
        """
        Sample a small subset of the local dataset to pass to trigger.train_trigger.
        Returns a DataLoader of clean (non-poisoned) samples.
        """
        base_dataset = self.trainloader.dataset
        N = len(base_dataset)
        if N == 0:
            raise RuntimeError("Local dataset empty; cannot train A3FL trigger")

        k = min(self.trigger_sample_size, N)
        # deterministic sampling using client id + epoch seed behaviour might be desirable
        rng = np.random.RandomState(int(self.get_id()) ^ 0xA5A5A5)
        indices = rng.choice(np.arange(N), size=k, replace=False).tolist()

        sampled_ds = Subset(base_dataset, indices)
        batch_size = min(getattr(self.trainloader, "batch_size", 32), k)
        return DataLoader(sampled_ds, batch_size=batch_size, shuffle=True, num_workers=getattr(self.trainloader, "num_workers", 0))

    def _build_poisoned_train_loader(self) -> DataLoader:
        """
        Create a backdoor training loader that mixes poisoned and clean samples.
        """
        base_dataset = self.trainloader.dataset
        batch_size = getattr(self.trainloader, "batch_size", 32)
        seed = int(self.get_id())

        # The create_backdoor_train_loader function takes the selector directly
        poisoned_loader = create_backdoor_train_loader(
            base_dataset=base_dataset,
            selector=self.selector,  
            trigger=self.trigger,
            target_class=self.target_class,
            batch_size=batch_size,
            shuffle=True,
        )
        return poisoned_loader

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        1) Optimize the trigger using local clean subset (A3FL adaptation).
        2) Train on poisoned+clean loader using BenignClient.local_train behavior
           (we temporarily swap self.trainloader to a poisoned loader).
        """
        # 0) Basic guards: no training if dataset empty
        if self.trainloader is None or len(self.trainloader.dataset) == 0:
            return super().local_train(epochs, round_idx)

        device = self.device if hasattr(self, "device") else torch.device("cpu")

        # 1) Train the trigger on a small subset of local clean data
        try:
            trigger_dl = self._build_trigger_dataloader()
            # train_trigger requires classifier_model, dataloader, target_class, epochs, learning_rate, lambda_balance
            # We pass a copy of the client's model to avoid poisoning the model weights during trigger optimization.
            # BUT A3FL's train_trigger will internally freeze/unfreeze classifier_model and DOES NOT update model weights
            # (it trains adversarial model internally). Still, to be safe we pass a deepcopy of the model.
            classifier_model = cp.deepcopy(self.model)
            classifier_model.to(device)
            # Move the trigger's pattern to correct device
            self.trigger.pattern.data = self.trigger.pattern.data.to(device)
            # Call the A3FL trigger training routine
            self.trigger.train_trigger(
                classifier_model=classifier_model,
                dataloader=trigger_dl,
                target_class=self.target_class,
                epochs=self.trigger_epochs,
                learning_rate=self.trigger_lr,
                lambda_balance=self.trigger_lambda_balance
            )
        except Exception as e:
            # if trigger training fails for any reason, log and continue with original trigger (fail-safe)
            print(f"[A3FLClient {self.get_id()}] trigger training failed (round {round_idx}): {e}")
            # proceed without updated trigger

        # 2) Build the poisoned loader and run local training on it
        poisoned_loader = None
        try:
            poisoned_loader = self._build_poisoned_train_loader()
        except Exception as e:
            print(f"[A3FLClient {self.get_id()}] failed to create poisoned loader, falling back to benign training: {e}")
            return super().local_train(epochs, round_idx)

        # Swap the client's trainloader to the poisoned one temporarily, run BenignClient.local_train, then restore
        orig_loader = self.trainloader
        try:
            self.trainloader = poisoned_loader
            # Use parent's local_train which uses self._model, optimizer, scheduler etc.
            result = super().local_train(epochs, round_idx)
        finally:
            # restore original loader
            self.trainloader = orig_loader

        # Optionally attach metadata about the trigger (e.g., pattern snapshot) for debugging
        try:
            # copy trigger pattern to CPU numpy for logging if desired
            pat = self.trigger.pattern.detach().cpu().numpy()
            result.setdefault("metadata", {})["trigger_pattern_snapshot"] = pat
        except Exception:
            pass

        return result
