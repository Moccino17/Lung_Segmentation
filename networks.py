import torch.nn as nn
from layers import *


class TestNet(nn.Module):
    def __init__(self, n=32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, n),
            ConvBlock(n, n * 2),
            ConvBlock(n * 2, n * 4),
            ConvBlock(n * 4, n * 8),
            DeconvBlock(n * 8, n * 4),
            DeconvBlock(n * 4, n * 2),
            DeconvBlock(n * 2, n),
            DeconvBlock(n, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)