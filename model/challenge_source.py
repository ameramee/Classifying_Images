"""
EECS 445 - Introduction to Machine Learning
Winter 2025 - Project 2

Source CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.source import Source
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_random_seed


__all__ = ["Source"]


class Csource(nn.Module):
    def __init__(self) -> None:
        """Define model architecture."""
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 5, stride = 2, padding = 2)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 2, padding = 2)
        self.fcone = nn.Linear(in_features = 256, out_features = 8)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights."""
        set_random_seed()

        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(conv.weight, mean = 0.0, std = 1 / ((5 * 5 * conv.in_channels)**0.5))
            nn.init.constant_(conv.bias, 0.0)
            

        nn.init.normal_(self.fcone.weight, mean = 0.0, std = 1 / ((self.fcone.in_features)**0.5))
        nn.init.constant_(self.fcone.bias, 0.0)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward propagation for a batch of input examples. Pass the input array
        through layers of the model and return the output after the final layer.

        Args:
            x: array of shape (N, C, H, W) 
                N = number of samples
                C = number of channels
                H = height
                W = width

        Returns:
            z: array of shape (1, # output classes)
        """
        N, C, H, W = x.shape
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fcone(x)

        return x
