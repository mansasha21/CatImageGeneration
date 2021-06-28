import torch
import torch.nn as nn

from src.utils.utils import init_weights

CHANNELS = [64, 128, 256, 512]
STRIDES = [2, 2, 2, 1]
IN_CHANNELS = 3


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=4):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.convblock(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS):
        super().__init__()
        convb0 = nn.Sequential(
            nn.Conv2d(in_channels * 2, CHANNELS[0], kernel_size=4, stride=STRIDES[0], padding_mode='reflect', bias=False),
            nn.LeakyReLU(0.2)
        )
        convblocks = [convb0]
        in_channels = CHANNELS[0]

        for channel, stride in zip(CHANNELS[1:], STRIDES[1:]):
            convblocks.append(ConvBlock(in_channels, channel, stride))
            in_channels = channel

        convblocks.append(nn.Conv2d(in_channels, 1, kernel_size=4, padding=1, padding_mode='reflect'))
        convblocks.append(nn.Sigmoid())
        self.model = nn.Sequential(*convblocks)
        self.model.apply(init_weights)

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))
