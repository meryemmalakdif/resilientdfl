import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN_CIFAR(nn.Module):
    """
    A simple but effective CNN architecture for 32x32 color images.
    Suitable for the CIFAR-10 dataset.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Input is 3x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # -> 32x16x16
        x = self.pool(F.relu(self.conv2(x))) # -> 64x8x8
        x = self.pool(F.relu(self.conv3(x))) # -> 64x4x4
        x = x.view(-1, 64 * 4 * 4)           # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x