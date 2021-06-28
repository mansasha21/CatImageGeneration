# Fully implemented by myself but inspired by Aladdin Persson (https://www.youtube.com/watch?v=9SGs4Nm0VR4)
# Я пробовал использовать Upsample вместо ConvTranspose2d, но результаты оставались на том же уровне или хуже

import torch
import torch.nn as nn

from src.utils.utils import init_weights


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, use_dropout=False, activation=nn.ReLU()):
        super().__init__()
        if downsample:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode='reflect', bias=False),
                nn.BatchNorm2d(out_channels),
                activation
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                activation
            )

        self.block.apply(init_weights)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, channels=64):
        super().__init__()

        self.down0 = nn.Sequential(
            nn.Conv2d(in_channels, channels, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        self.down0.apply(init_weights)

        self.down1 = UnetBlock(channels, channels * 2, activation=nn.LeakyReLU(0.2))
        self.down2 = UnetBlock(channels * 2, channels * 4, activation=nn.LeakyReLU(0.2))
        self.down3 = UnetBlock(channels * 4, channels * 8, activation=nn.LeakyReLU(0.2))
        self.down4 = UnetBlock(channels * 8, channels * 8, activation=nn.LeakyReLU(0.2))
        self.down5 = UnetBlock(channels * 8, channels * 8, activation=nn.LeakyReLU(0.2))
        self.down6 = UnetBlock(channels * 8, channels * 8, activation=nn.LeakyReLU(0.2))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels * 8, channels * 8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU()
        )
        self.bottleneck.apply(init_weights)

        self.up0 = UnetBlock(channels * 8, channels * 8, downsample=False, use_dropout=True)
        self.up1 = UnetBlock(channels * 16, channels * 8, downsample=False, use_dropout=True)
        self.up2 = UnetBlock(channels * 16, channels * 8, downsample=False, use_dropout=True)
        self.up3 = UnetBlock(channels * 16, channels * 8, downsample=False)
        self.up4 = UnetBlock(channels * 16, channels * 4, downsample=False)
        self.up5 = UnetBlock(channels * 8, channels * 2, downsample=False)
        self.up6 = UnetBlock(channels * 4, channels, downsample=False)

        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, in_channels, 4, 2, 1),
            nn.Tanh()
        )
        self.up7.apply(init_weights)

    def forward(self, x):
        d1 = self.down0(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)

        bn = self.bottleneck(d7)

        up1 = self.up0(bn)
        up2 = self.up1(torch.cat([up1, d7], 1))
        up3 = self.up2(torch.cat([up2, d6], 1))
        up4 = self.up3(torch.cat([up3, d5], 1))
        up5 = self.up4(torch.cat([up4, d4], 1))
        up6 = self.up5(torch.cat([up5, d3], 1))
        up7 = self.up6(torch.cat([up6, d2], 1))
        return self.up7(torch.cat([up7, d1], 1))
