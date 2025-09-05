import torch
import torch.nn as nn
import torch.nn.functional as F

class GTSRB_CNN(nn.Module):
    """
    A CNN architecture designed for the German Traffic Sign Recognition Benchmark.
    Handles 3-channel images and 43 classes. Assumes input is resized to 32x32.
    """
    def __init__(self, num_classes=43):
        super().__init__()
        # Input is 3x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # -> 32x14x14
        x = self.pool(F.relu(self.conv2(x))) # -> 64x5x5
        x = x.view(-1, 64 * 5 * 5)           # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
