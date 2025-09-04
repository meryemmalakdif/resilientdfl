from typing import Any, Dict, Set
import torch
from torch.utils.data import DataLoader

# Import from your project structure
from ..fl.baseclient import BenignClient
from ..datasets.backdoor import PoisonedDataset
from .selectors.base import BaseSelector
from .triggers.base import BaseTrigger
from .poisoners.base import BasePoisoner

class MaliciousClient(BenignClient):
    """
    A malicious client that inherits from BenignClient and overrides the
    local training method to perform a backdoor attack.
    """
    def __init__(
        self,
        # Standard BenignClient arguments
        id: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        epochs: int,
        device: torch.device,
        # Malicious components
        selector: BaseSelector,
        trigger: BaseTrigger,
        poisoner: BasePoisoner,
        target_class: int
    ):
        # Initialize the parent BenignClient
        super().__init__(id, trainloader, testloader, model, lr, weight_decay, epochs, device)

        # Store the attack components
        self.selector = selector
        self.trigger = trigger
        self.poisoner = poisoner
        self.target_class = target_class

    def local_train(self, epochs: int, round_idx: int, **kwargs) -> Dict[str, Any]:
        """
        Overrides the benign training process to inject a backdoor.
        """
        print(f"\n--- Malicious Client [{self.get_id()}] starting training for round {round_idx} ---")

        # 1. Update trigger if it's trainable (e.g., A3FL, IBA)
        if hasattr(self.trigger, 'train_trigger'):
            print(f"Client [{self.get_id()}]: Updating A3FL trigger...")
            # This requires access to the full dataloader
            full_dataloader = DataLoader(self.trainloader.dataset, batch_size=self.trainloader.batch_size, shuffle=True)
            self.trigger.train_trigger(self.model, full_dataloader, self.target_class)

        # 2. Select samples to poison for this round
        poisoned_indices = self.selector.select(self.trainloader.dataset)
        print(f"Client [{self.get_id()}]: Selected {len(poisoned_indices)} samples to poison.")

        # 3. Create a poisoned view of the dataset for training
        poisoned_dataset = PoisonedDataset(
            original_dataset=self.trainloader.dataset,
            poisoned_indices=set(poisoned_indices),
            trigger=self.trigger,
            target_class=self.target_class
        )
        poisoned_dataloader = DataLoader(poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)

        # 4. Use the specified Poisoner to train the model
        print(f"Client [{self.get_id()}]: Handing off to {self.poisoner.__class__.__name__}.")
        self._model = self.poisoner.poison(
            model=self.model,
            dataloader=poisoned_dataloader,
            poisoned_indices=set(poisoned_indices),
            epochs=epochs if epochs is not None else self.epochs_default,
            learning_rate=self.lr,
            device=self.device,
            **kwargs  # Pass extra args like prev_global_grad to the poisoner
        )
        
        # 5. Return results in the format expected by the server
        metrics = self.local_evaluate()['metrics']
        result = {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }
        print(f"--- Malicious Client [{self.get_id()}] finished training. ---")
        return result