from .base import BaseTrigger
from ...models import UNet  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class IBATrigger(BaseTrigger):
    """
    Implements the generative trigger from "IBA", now including the
    ability to train the U-Net generator against a specific classifier.
    """
    def __init__(self, unet_model: UNet, alpha: float = 0.2):
        self.generator = unet_model
        self.generator.eval()
        super().__init__(position=(0, 0), size=(0, 0), pattern=self.generator, alpha=alpha)

    def _freeze_model(self, model):
        """Helper function to freeze model parameters."""
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_model(self, model):
        """Helper function to unfreeze model parameters."""
        for param in model.parameters():
            param.requires_grad = True

    def train_generator(self, classifier_model, dataloader: DataLoader, target_class: int, 
                        epochs: int = 5, learning_rate: float = 1e-3):
        """
        Trains the U-Net generator to fool a given classifier model.

        Args:
            classifier_model: The target model to attack (e.g., the global model).
            dataloader (DataLoader): The client's local data loader.
            target_class (int): The target label for the backdoor attack.
            epochs (int): Number of epochs to train the generator.
            learning_rate (float): Learning rate for the generator's optimizer.
        """
        device = next(classifier_model.parameters()).device
        self.generator.to(device)
        
        # 1. Freeze the classifier model and set it to evaluation mode
        classifier_model.eval()
        self._freeze_model(classifier_model)

        # 2. Set the U-Net generator to training mode
        self.generator.train()
        
        optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        print(f"--- Training IBA Generator for {epochs} epochs ---")
        for epoch in range(epochs):
            total_loss = 0
            correct_preds = 0
            total_samples = 0

            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                
                optimizer.zero_grad()
                
                # 3. Generate poisoned inputs using the U-Net
                perturbation = self.generator(inputs)
                poisoned_inputs = torch.clamp(inputs + self.alpha * perturbation, 0.0, 1.0)
                
                # Create target labels
                poisoned_labels = torch.full((inputs.size(0),), target_class, dtype=torch.long, device=device)
                
                # 4. Get the classifier's output on the poisoned data
                outputs = classifier_model(poisoned_inputs)
                
                # 5. Calculate loss and backpropagate *only to the U-Net*
                loss = loss_fn(outputs, poisoned_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct_preds += (preds == poisoned_labels).sum().item()
                total_samples += inputs.size(0)
            
            epoch_loss = total_loss / len(dataloader)
            epoch_asr = correct_preds / total_samples
            print(f"Epoch {epoch+1}/{epochs} | Generator Loss: {epoch_loss:.4f} | Attack Success Rate: {epoch_asr:.4f}")

        # 6. Unfreeze the classifier and set the generator back to eval mode
        self._unfreeze_model(classifier_model)
        self.generator.eval()
        print("--- Generator training finished ---")

    @torch.no_grad()
    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the generative trigger to a clean image. (Unchanged from before)
        """
        device = image.device
        self.pattern.to(device)
        self.pattern.eval() # Ensure generator is in eval mode for inference
        
        image_batch = image.unsqueeze(0)
        perturbation = self.pattern(image_batch)
        perturbation = perturbation.squeeze(0)
        
        poisoned_image = torch.clamp(image + self.alpha * perturbation, 0.0, 1.0)
        
        return poisoned_image