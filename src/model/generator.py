import torch
import torch.nn as nn

from src.utils.utils import init_weights


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True, use_dropout=False, activation=nn.ReLU()):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode='reflect', bias=False)
            if downsample
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
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
    def __init__(self, in_channels=3, filters=64):
        super().__init__()

        self.down0 = nn.Sequential(
            nn.Conv2d(in_channels, filters, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        self.down0.apply(init_weights)

        self.down1 = UnetBlock(filters, filters*2, activation=nn.LeakyReLU(0.2))
        self.down2 = UnetBlock(filters*2, filters*4, activation=nn.LeakyReLU(0.2))
        self.down3 = UnetBlock(filters*4, filters*8, activation=nn.LeakyReLU(0.2))
        self.down4 = UnetBlock(filters*8, filters*8, activation=nn.LeakyReLU(0.2))
        self.down5 = UnetBlock(filters*8, filters*8, activation=nn.LeakyReLU(0.2))
        self.down6 = UnetBlock(filters*8, filters*8, activation=nn.LeakyReLU(0.2))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(filters*8, filters*8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU()
        )
        self.bottleneck.apply(init_weights)

        self.up0 = UnetBlock(filters*8, filters*8, downsample=False, use_dropout=True)
        self.up1 = UnetBlock(filters*16, filters*8, downsample=False, use_dropout=True)
        self.up2 = UnetBlock(filters*16, filters*8, downsample=False, use_dropout=True)
        self.up3 = UnetBlock(filters*16, filters*8, downsample=False,)
        self.up4 = UnetBlock(filters*16, filters*4, downsample=False,)
        self.up5 = UnetBlock(filters*8, filters*2, downsample=False,)
        self.up6 = UnetBlock(filters*4, filters, downsample=False)

        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(filters*2, in_channels, 4, 2, 1),
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

