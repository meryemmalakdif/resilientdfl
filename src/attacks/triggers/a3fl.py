import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, position=(28, 28), size=(5, 5), alpha=1.0, in_channels: int = 3,
                 # Optimization parameters from the paper/repo
                 trigger_epochs: int = 100,
                 trigger_lr: float = 0.01,
                 lambda_balance: float = 0.1,
                 adv_epochs: int = 2,
                 adv_lr: float = 0.01):
        """
        Initializes the A3FL trigger with all configurable parameters.
        """
        initial_pattern = torch.rand(in_channels, size[1], size[0], requires_grad=True)
        self.pattern = initial_pattern
        super().__init__(position, size, self.pattern, alpha=alpha)

        self.trigger_epochs = trigger_epochs
        self.trigger_lr = trigger_lr
        self.lambda_balance = lambda_balance
        self.adv_epochs = adv_epochs
        self.adv_lr = adv_lr


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
        local_pattern = self.pattern.to(images.device)
        blend_pattern = self.alpha * local_pattern.unsqueeze(0) + (1.0 - self.alpha) * region
        poisoned_images[:, :, y_start:y_end, x_start:x_end] = blend_pattern
        return torch.clamp(poisoned_images, 0.0, 1.0)

    def _get_adversarial_model(self, classifier_model, dataloader):
        """Creates the hardened adversarial model by training it to unlearn the trigger."""
        adv_model = copy.deepcopy(classifier_model)
        self._unfreeze_model(adv_model)
        adv_model.train()
        
        device = next(adv_model.parameters()).device
        optimizer = optim.SGD(adv_model.parameters(), lr=self.adv_lr) # Use self.adv_lr
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.adv_epochs): # Use self.adv_epochs
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                poisoned_inputs = self._apply_batch(inputs)
                optimizer.zero_grad()
                outputs = adv_model(poisoned_inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
        self._freeze_model(adv_model)
        adv_model.eval()
        return adv_model

    def _get_model_similarity(self, model_a: nn.Module, model_b: nn.Module) -> torch.Tensor:
        """Helper to compute cosine similarity between the parameters of two models."""
        model_a.eval()
        model_b.eval()
        
        params_a = torch.cat([p.view(-1) for p in model_a.parameters()])
        params_b = torch.cat([p.view(-1) for p in model_b.parameters()])
        
        params_b = params_b.to(params_a.device)
        return F.cosine_similarity(params_a, params_b, dim=0, eps=1e-8)

    def train_trigger(self, classifier_model, dataloader: DataLoader, target_class: int):
        """
        Optimizes the trigger patch using the A3FL adversarial adaptation process.
        """
        device = next(classifier_model.parameters()).device
        self.pattern = self.pattern.to(device).requires_grad_(True)
        
        print("--- Training A3FL Trigger using PGD ---")
        adversarial_model = self._get_adversarial_model(classifier_model, dataloader)
        
        self._freeze_model(classifier_model)
        classifier_model.eval()

        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(self.trigger_epochs): # Use self.trigger_epochs
            total_loss, backdoor_loss_val, adaptation_loss_val = 0, 0, 0
            
            similarity = self._get_model_similarity(classifier_model, adversarial_model)
            dynamic_lambda = self.lambda_balance * similarity # Use self.lambda_balance
            
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                self.pattern.requires_grad = True

                poisoned_inputs = self._apply_batch(inputs)
                poisoned_labels = torch.full((inputs.size(0),), target_class, dtype=torch.long, device=device)

                outputs_orig = classifier_model(poisoned_inputs)
                backdoor_loss = loss_fn(outputs_orig, poisoned_labels)
                
                outputs_adv = adversarial_model(poisoned_inputs)
                adaptation_loss = loss_fn(outputs_adv, poisoned_labels)
                
                total_loss_batch = backdoor_loss + dynamic_lambda * adaptation_loss
                
                if self.pattern.grad is not None:
                    self.pattern.grad.zero_()

                total_loss_batch.backward()
                
                with torch.no_grad():
                    # Use self.trigger_lr for the PGD step
                    self.pattern.sub_(self.trigger_lr * self.pattern.grad.sign())
                    self.pattern.clamp_(0, 1)

                total_loss += total_loss_batch.item()
                backdoor_loss_val += backdoor_loss.item()
                adaptation_loss_val += adaptation_loss.item()

            print(f"Epoch {epoch+1}/{self.trigger_epochs} | Sim: {similarity:.4f} | Dyn λ: {dynamic_lambda:.4f} | Total Loss: {total_loss/len(dataloader):.4f}")

        self._unfreeze_model(classifier_model)
        print("--- A3FL Trigger training finished ---")

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the optimized trigger patch to a single image tensor of shape (C, H, W).
        """
        if image.dim() != 3:
            raise ValueError(f"The `apply` method expects a single 3D image tensor (C, H, W), but got shape {image.shape}")

        poisoned_image = image.clone()
        _, img_h, img_w = poisoned_image.shape
        patch_w, patch_h = self.size
        x_pos, y_pos = self.position
        pattern = self.pattern.to(image.device)
        y_start = min(y_pos, img_h)
        y_end = min(y_pos + patch_h, img_h)
        x_start = min(x_pos, img_w)
        x_end = min(x_pos + patch_w, img_w)
        region = poisoned_image[:, y_start:y_end, x_start:x_end]
        blend_pattern = self.alpha * pattern + (1.0 - self.alpha) * region
        poisoned_image[:, y_start:y_end, x_start:x_end] = blend_pattern
        return torch.clamp(poisoned_image, 0, 1)

