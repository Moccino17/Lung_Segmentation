import torch.nn as nn
from layers import *


class TestNet(nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, n),
            ConvBlock(n, n * 2),
            ConvBlock(n * 2, n * 4),
            ConvBlock(n * 4, n * 8),
            DeconvBlock(n * 8, n * 4),
            DeconvBlock(n * 4, n * 2),
            DeconvBlock(n * 2, n),
            DeconvBlock(n, 1)
        )

    def forward(self, x):
        return self.net(x)


class TestNet2(nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, n, kernel_size=3, padding=1, bias=False, stride=2),
            nn.ConvTranspose2d(n, 1, kernel_size=2, stride=2, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)