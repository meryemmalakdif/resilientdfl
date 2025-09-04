import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from .base import BaseTrigger

class A3FLTrigger(BaseTrigger):
    """
    Implements the adversarially adaptive trigger from "A3FL".

    The trigger is a learnable tensor (a patch) that is optimized
    against both the current global model and a simulated "adversarial" model
    that has been trained to unlearn the backdoor.
    """
    def __init__(self, position=(28, 28), size=(5, 5), alpha=1.0):
        """
        Initializes the A3FL trigger.

        Args:
            position (tuple): Top-left (x, y) coordinates of the trigger.
            size (tuple): (width, height) of the trigger patch.
            alpha (float): The blending factor for applying the trigger.
        """
        # The pattern starts as a random tensor and is made a learnable parameter.
        # It's shaped (C, H, W) for a color image.
        initial_pattern = torch.rand(3, size[1], size[0])
        self.pattern = nn.Parameter(initial_pattern, requires_grad=True)

        super().__init__(position, size, self.pattern, alpha=alpha)

    def _freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def _apply_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Helper to apply the trigger to a batch of images."""
        poisoned_images = images.clone()
        _, _, img_h, img_w = images.shape
        patch_w, patch_h = self.size
        x_pos, y_pos = self.position

        y_start = min(y_pos, img_h)
        y_end = min(y_pos + patch_h, img_h)
        x_start = min(x_pos, img_w)
        x_end = min(x_pos + patch_w, img_w)
        
        region = poisoned_images[:, :, y_start:y_end, x_start:x_end]
        
        # Blend the patch using the alpha factor
        blend_pattern = self.alpha * self.pattern + (1.0 - self.alpha) * region
        poisoned_images[:, :, y_start:y_end, x_start:x_end] = blend_pattern
        
        return torch.clamp(poisoned_images, 0.0, 1.0)

    def _get_adversarial_model(self, classifier_model, dataloader, adv_epochs=2, adv_lr=0.01):
        """Creates the hardened adversarial model by training it to unlearn the trigger."""
        adv_model = copy.deepcopy(classifier_model)
        self._unfreeze_model(adv_model)
        adv_model.train()
        
        optimizer = optim.SGD(adv_model.parameters(), lr=adv_lr)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(adv_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.pattern.device), labels.to(self.pattern.device)
                
                # Poison inputs with the current trigger
                poisoned_inputs = self._apply_batch(inputs)
                
                optimizer.zero_grad()
                outputs = adv_model(poisoned_inputs)
                # Train on ground-truth labels to "unlearn" the backdoor
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
        self._freeze_model(adv_model)
        adv_model.eval()
        return adv_model

    def train_trigger(self, classifier_model, dataloader: DataLoader, target_class: int,
                      epochs: int = 10, learning_rate: float = 0.01, lambda_balance: float = 0.1):
        """
        Optimizes the trigger patch using the A3FL adversarial adaptation process.
        """
        device = next(classifier_model.parameters()).device
        self.pattern.data = self.pattern.data.to(device) # Move trigger to the correct device
        
        print("--- Training A3FL Trigger ---")
        # 1. Create the hardened adversarial model
        print("Creating adversarial model...")
        adversarial_model = self._get_adversarial_model(classifier_model, dataloader)
        
        # 2. Freeze the main classifier
        self._freeze_model(classifier_model)
        classifier_model.eval()

        # 3. Optimize the trigger patch
        optimizer = optim.Adam([self.pattern], lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss, backdoor_loss_val, adaptation_loss_val = 0, 0, 0
            
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                poisoned_inputs = self._apply_batch(inputs)
                poisoned_labels = torch.full((inputs.size(0),), target_class, dtype=torch.long, device=device)

                optimizer.zero_grad()
                
                # Loss against the original model
                outputs_orig = classifier_model(poisoned_inputs)
                backdoor_loss = loss_fn(outputs_orig, poisoned_labels)
                
                # Loss against the adversarial model
                outputs_adv = adversarial_model(poisoned_inputs)
                adaptation_loss = loss_fn(outputs_adv, poisoned_labels)
                
                # Combine losses as per the A3FL paper
                total_loss_batch = backdoor_loss + lambda_balance * adaptation_loss
                total_loss_batch.backward()
                optimizer.step()

                # Project trigger values to be in the valid [0, 1] range
                self.pattern.data.clamp_(0, 1)

                total_loss += total_loss_batch.item()
                backdoor_loss_val += backdoor_loss.item()
                adaptation_loss_val += adaptation_loss.item()

            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss/len(dataloader):.4f} "
                  f"| Backdoor Loss: {backdoor_loss_val/len(dataloader):.4f} "
                  f"| Adaptation Loss: {adaptation_loss_val/len(dataloader):.4f}")

        self._unfreeze_model(classifier_model)
        print("--- A3FL Trigger training finished ---")

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """Applies the optimized trigger patch to a single image."""
        poisoned_image = image.clone()
        _, img_h, img_w = poisoned_image.shape
        patch_w, patch_h = self.size
        x_pos, y_pos = self.position

        # Move the pattern to the same device as the image tensor
        pattern = self.pattern.to(image.device)

        y_start = min(y_pos, img_h)
        y_end = min(y_pos + patch_h, img_h)
        x_start = min(x_pos, img_w)
        x_end = min(x_pos + patch_w, img_w)

        region = poisoned_image[:, y_start:y_end, x_start:x_end]
        
        # Blend the patch
        blend_pattern = self.alpha * pattern + (1.0 - self.alpha) * region
        poisoned_image[:, y_start:y_end, x_start:x_end] = blend_pattern

        return torch.clamp(poisoned_image, 0, 1)