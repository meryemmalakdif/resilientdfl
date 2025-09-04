import torch
from torch.utils.data import Dataset
from typing import Set, List

# Assuming your trigger base class is accessible
# from ..attacks.triggers import BaseTrigger

class PoisonedDataset(Dataset):
    """
    A wrapper dataset to apply triggers and modify labels for selected samples.
    
    This is used by the malicious client to create a poisoned view of its
    local data without modifying the original dataset.
    """
    def __init__(self, 
                 original_dataset: Dataset, 
                 poisoned_indices: Set[int], 
                 trigger,  # BaseTrigger
                 target_class: int):
        self.original_dataset = original_dataset
        self.poisoned_indices = poisoned_indices
        self.trigger = trigger
        self.target_class = target_class

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        # Retrieve the original image and label
        image, label = self.original_dataset[index]
        
        # If the index is marked for poisoning, apply the attack
        if index in self.poisoned_indices:
            image = self.trigger.apply(image)
            label = self.target_class
            
        return image, label