import torch.nn
import torch.nn as nn
from torchvision.models import efficientnet_b4
from torchsummary import summary


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Dropout(0.2),
            ResBlock(out_channels, out_channels),
            ResBlock(out_channels, out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.skip_connection = nn.Sequential(
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return self.skip_connection(x) + self.main(x)


class UpBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Dropout(0.2),
            ResBlock(out_channels, out_channels),
            ResBlock(out_channels, out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.main(x)


class BaselineNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = efficientnet_b4(weights='IMAGENET1K_V1')
        self.middle_block = UpBlock(1792, 512)
        self.up_block1 = UpBlock(672, 312)
        self.up_block2 = UpBlock(368, 128)
        self.up_block3 = UpBlock(160, 64)
        self.up_block4 = UpBlock(88, 32)
        self.output_block = nn.Sequential(
            ConvBlock(32, 16),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        f128 = features[1]
        f64 = features[2]
        f32 = features[3]
        f16 = features[5]
        y = features[-1]
        y = self.middle_block(y)
        y = self.up_block1(torch.cat((f16, y), dim=1))
        y = self.up_block2(torch.cat((f32, y), dim=1))
        y = self.up_block3(torch.cat((f64, y), dim=1))
        y = self.up_block4(torch.cat((f128, y), dim=1))
        return self.output_block(y)


if __name__ == '__main__':
    # model = efficientnet_b4(weights='IMAGENET1K_V1').cuda()
    model = BaselineNetwork().cuda()
    summary(model, (3, 256, 256))