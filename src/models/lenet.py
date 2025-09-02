import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=62):  # adjust classes for FEMNIST
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # input 1x28x28
        self.pool  = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # -> 6x24x24
        x = self.pool(x)             # -> 6x12x12
        x = F.relu(self.conv2(x))    # -> 16x8x8
        x = self.pool(x)             # -> 16x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x