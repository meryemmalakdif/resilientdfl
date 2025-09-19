import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
# MODIFICATION: Import Rectangle for visualization
from matplotlib.patches import Rectangle

# --- Import Framework Components ---
from src.datasets.mnist import MNISTAdapter
from src.models.lenet import LeNet5
from src.attacks.triggers.a3fl import A3FLTrigger

def visualize_a3fl_trigger(config):
    """
    Trains an A3FL trigger for a few epochs and visualizes its effect on a sample image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. --- Setup Dataset and Model ---
    dataset_adapter = MNISTAdapter(root=config['data_root'], download=True)
    test_loader = dataset_adapter.get_test_loader(batch_size=config['batch_size'])
    train_loader_for_trigger = dataset_adapter.get_test_loader(batch_size=config['batch_size'], shuffle=True)

    model = LeNet5(num_classes=10).to(device)

    # 2. --- Instantiate the A3FL Trigger ---
    # MODIFICATION: Pass the position from the config
    a3fl_trigger = A3FLTrigger(
        position=config['position'],
        size=(config['trigger_size'], config['trigger_size']),
        in_channels=1, # MNIST is grayscale
        trigger_epochs=config['trigger_epochs'],
        trigger_lr=config['trigger_lr']
    )

    # 3. --- Train the Trigger ---
    print(f"Training the A3FL trigger for {config['trigger_epochs']} epochs...")
    a3fl_trigger.train_trigger(
        classifier_model=model,
        dataloader=train_loader_for_trigger,
        target_class=config['target_class']
    )
    print("Trigger training complete.")

    # 4. --- Get a Sample Image ---
    original_image, original_label = test_loader.dataset[config['image_index']]
    if not isinstance(original_image, torch.Tensor):
        original_image = torch.tensor(original_image)
    if original_image.dim() == 3:
         original_image = original_image.unsqueeze(0)

    # 5. --- Apply the Trigger ---
    poisoned_image_tensor = a3fl_trigger.apply(original_image.squeeze(0))

    # 6. --- Visualize the Results ---
    original_np = original_image.squeeze().cpu().numpy()
    poisoned_np = poisoned_image_tensor.squeeze().cpu().numpy()
    trigger_pattern_np = a3fl_trigger.pattern.detach().squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"A3FL Trigger Visualization (Target Class: {config['target_class']})", fontsize=16)

    axs[0].imshow(original_np, cmap='gray')
    axs[0].set_title(f"Original Image (Label: {original_label})")
    axs[0].axis('off')

    axs[1].imshow(trigger_pattern_np, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Optimized Trigger Pattern")
    axs[1].axis('off')

    axs[2].imshow(poisoned_np, cmap='gray')
    axs[2].set_title("Image with Trigger Applied")
    axs[2].axis('off')
    
    # MODIFICATION: Add a rectangle to show the trigger's position
    rect = Rectangle(
        (config['position'][0] - 0.5, config['position'][1] - 0.5), # xy bottom-left corner
        config['trigger_size'], config['trigger_size'],             # width, height
        linewidth=1.5, edgecolor='lime', facecolor='none'
    )
    axs[2].add_patch(rect)


    output_path = "a3fl_visualization.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize A3FL Trigger")
    # MODIFICATION: Add position argument with a safe default for MNIST (28x28)
    parser.add_argument("--position", type=int, nargs=2, default=[23, 23], help="Top-left (x, y) coords of the trigger patch.")
    parser.add_argument("--image_index", type=int, default=15, help="Index of the test image to visualize.")
    parser.add_argument("--trigger_size", type=int, default=5, help="Size of the trigger patch (e.g., 5 for 5x5).")
    parser.add_argument("--target_class", type=int, default=7, help="Target class for the backdoor.")
    parser.add_argument("--trigger_epochs", type=int, default=5, help="Number of epochs to train the trigger.")
    parser.add_argument("--trigger_lr", type=float, default=0.01, help="Learning rate for trigger optimization.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for trigger training.")
    parser.add_argument("--data_root", type=str, default="data", help="Root directory for datasets.")
    args = parser.parse_args()
    
    config = vars(args)
    visualize_a3fl_trigger(config)

