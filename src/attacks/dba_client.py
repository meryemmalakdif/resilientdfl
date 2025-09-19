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
        attack_start_round: int,
        attack_end_round: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(trigger, DBATrigger):
            raise TypeError("DBAClient must be initialized with a DBATrigger instance.")
        self.selector = selector
        self.trigger = trigger
        self.target_class = target_class
        self.poisoned_dataloader = create_backdoor_train_loader(
            base_dataset=self.trainloader.dataset,
            selector=self.selector,
            trigger=self.trigger,
            target_class=self.target_class,
            batch_size=self.trainloader.batch_size,
            shuffle=True
        )
        self.attack_start_round = attack_start_round
        self.attack_end_round = attack_end_round

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """
        Overrides the benign training process to perform a DBA attack.
        """
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super().local_train(epochs, round_idx)
        

        # Perform standard training on the poisoned dataloader
        train_loss, correct, total = 0.0, 0, 0
        
        for _ in range(epochs or self.epochs_default):
            if self.poisoned_dataloader is None: break
            for inputs, targets in self.poisoned_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        if self.scheduler:
            self.scheduler.step()

        num_batches = len(self.trainloader) if self.trainloader else 1
        avg_loss = train_loss / (num_batches * (epochs or self.epochs_default))
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {'loss': avg_loss, 'accuracy': accuracy}
        
        result = {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }
        return result


# from typing import Any, Dict

# from .scaling_client import ScalingAttackClient
# from .selectors.base import BaseSelector
# from .triggers.distributed import DBATrigger


# class DBAClient(ScalingAttackClient):
#     """
#     A malicious client for the Distributed Backdoor Attack (DBA).

#     This client implements the single-shot version of the DBA attack, which
#     combines a distributed trigger with model scaling. It inherits the core
#     training and scaling logic from ScalingAttackClient.
#     """
#     def __init__(
#         self,
#         selector: BaseSelector,
#         trigger: DBATrigger,
#         target_class: int,
#         **kwargs,
#     ):
#         # Ensure the trigger is of the correct type for this client
#         if not isinstance(trigger, DBATrigger):
#             raise TypeError("DBAClient must be initialized with a DBATrigger instance.")
        
#         # Call the parent constructor (ScalingAttackClient) with all arguments
#         super().__init__(
#             selector=selector,
#             trigger=trigger,
#             target_class=target_class,
#             **kwargs
#         )
