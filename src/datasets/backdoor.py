import torch
from torch.utils.data import Dataset, DataLoader
from typing import Set

from ..attacks.selectors.base import BaseSelector
from ..attacks.triggers.base import BaseTrigger

class PoisonedDataset(Dataset):
    """
    A Dataset wrapper that applies a trigger to specified indices on the fly.
    
    This is the core component for creating both poisoned training sets (for clients)
    and fully poisoned test sets (for ASR evaluation).
    """
    def __init__(self, 
                 original_dataset: Dataset, 
                 poisoned_indices: Set[int], 
                 trigger: BaseTrigger,  
                 target_class: int):
        self.original_dataset = original_dataset
        self.poisoned_indices = poisoned_indices
        self.trigger = trigger
        self.target_class = target_class

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        """
        Returns the sample at the given index. If the index is in the
        poisoned set, it applies the trigger and replaces the label.
        """
        image, label = self.original_dataset[index]
        
        if index in self.poisoned_indices:
            image = self.trigger.apply(image)
            label = self.target_class
            
        return image, label

def create_backdoor_train_loader(
    base_dataset: Dataset,
    selector: BaseSelector, 
    trigger: BaseTrigger,  
    target_class: int,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """
    Creates a DataLoader for training a malicious client.
    
    It uses a Selector to choose a subset of the base_dataset to poison.
    
    Args:
        base_dataset: The client's original, clean local dataset.
        selector: The selector object to choose which samples to poison.
        trigger: The trigger object to apply to the samples.
        target_class: The label to assign to poisoned samples.
        batch_size: The batch size for the DataLoader.
        shuffle: Whether to shuffle the DataLoader.
        
    Returns:
        A DataLoader that yields a mix of clean and poisoned data.
    """
    poisoned_indices = selector.select(base_dataset)
    poisoned_dataset = PoisonedDataset(
        original_dataset=base_dataset,
        poisoned_indices=set(poisoned_indices),
        trigger=trigger,
        target_class=target_class
    )
    return DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=shuffle)

def create_asr_test_loader(
    base_dataset: Dataset,
    trigger,  # BaseTrigger
    target_class: int,
    batch_size: int
) -> DataLoader:
    """
    Creates a DataLoader for evaluating the Attack Success Rate (ASR).
    
    It poisons ALL samples in the base_dataset (typically the test set).
    
    Args:
        base_dataset: The dataset to use for evaluation (e.g., test_loader.dataset).
        trigger: The trigger object to apply to all samples.
        target_class: The expected label for all triggered samples.
        batch_size: The batch size for the DataLoader.
        
    Returns:
        A DataLoader that yields a fully poisoned dataset.
    """
    # To poison all samples, we create a set of all indices
    all_indices = set(range(len(base_dataset)))
    poisoned_test_dataset = PoisonedDataset(
        original_dataset=base_dataset,
        poisoned_indices=all_indices,
        trigger=trigger,
        target_class=target_class
    )
    return DataLoader(poisoned_test_dataset, batch_size=batch_size, shuffle=False)
