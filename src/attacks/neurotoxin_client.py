from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader

from ..fl.baseclient import BenignClient
from ..datasets.backdoor import make_triggered_loader
from .selectors.base import BaseSelector
from .triggers.base import BaseTrigger

class NeurotoxinClient(BenignClient):
    """
    A malicious client for the Neurotoxin attack.

    This client behaves benignly until a specified start round, after which
    it poisons data and applies gradient constraints to make the backdoor
    more durable, as described in the Neurotoxin paper.
    """
    def __init__(
        self,
        # New parameters for the attack
        attack_start_round: int,
        # Attack specific components
        selector: BaseSelector,
        trigger: BaseTrigger,
        target_class: int,
        mask_k_percent: float = 0.05,
        # Pass all BenignClient arguments via kwargs
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.selector = selector
        self.trigger = trigger
        self.target_class = target_class
        self.attack_start_round = attack_start_round
        self.mask_k_percent = mask_k_percent
        print(f"NeurotoxinClient created. Will attack from round {attack_start_round} onward.")

    def local_train(self, epochs: int, round_idx: int, prev_global_grad: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Overrides the training process to perform a constrained attack.
        """
        # Behave benignly before the attack starts
        if round_idx < self.attack_start_round:
            print(f"--- Neurotoxin Client [{self.get_id()}] acting benignly in round {round_idx} ---")
            return super().local_train(epochs, round_idx)

        # On attack rounds, if we don't have a previous gradient (e.g., first attack round),
        # also behave benignly as we cannot compute the mask.
        if prev_global_grad is None:
            print(f"--- Neurotoxin Client [{self.get_id()}] acting benignly in round {round_idx} (no prev_global_grad) ---")
            return super().local_train(epochs, round_idx)

        print(f"--- Neurotoxin Client [{self.get_id()}] EXECUTING ATTACK in round {round_idx} ---")
        
        # 1. Determine importance mask from the previous global gradient
        all_grads = torch.cat([v.flatten().abs() for v in prev_global_grad.values()])
        threshold = torch.quantile(all_grads, 1.0 - self.mask_k_percent)
        importance_mask = {
            name: (grad.abs() >= threshold).to(self.device)
            for name, grad in prev_global_grad.items()
        }
        print(f"Client [{self.get_id()}]: Calculated importance threshold: {threshold.item():.4f}")

        # 2. Create a poisoned dataloader
        poisoned_dataloader = make_triggered_loader(
            base_dataset=self.trainloader.dataset,
            trigger=self.trigger,
            keep_label=False,
            forced_label=self.target_class,
            fraction=self.selector.poisoning_rate,
            batch_size=self.trainloader.batch_size,
            shuffle=True
        )

        # 3. Perform constrained training
        self._model.train()
        epoch_count = epochs if epochs is not None else self.epochs_default
        for _ in range(epoch_count):
            for inputs, targets in poisoned_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()

                # Apply the Neurotoxin constraint before the optimizer step
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in importance_mask:
                            param.grad[importance_mask[name]] = 0.0
                
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
