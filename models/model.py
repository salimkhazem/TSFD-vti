"""
UNet model implementation
"""

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    Returns:
        torch.Tensor: output tensor
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        bilinear (bool): whether to use bilinear interpolation
        or transposed convolutions
    Returns:
        torch.Tensor: output tensor
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions
        # to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffx // 2, diffx - diffx // 2, diffy // 2, diffy - diffy // 2],
        )
        # if you have padding issues, see below
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final convolution layer
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    Returns:
        torch.Tensor: output tensor
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Full assembly of the parts to form the complete UNet model
    Args:
        n_channels (int): number of input channels
        n_classes (int): number of output channels
        bilinear (bool): whether to use bilinear interpolation
        or transposed convolutions
    Returns:
        torch.Tensor: output tensor


    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    model = UNet(n_channels=1, n_classes=1)
    x = torch.randn((1, 1, 256, 256))
    out = model(x)
    assert (
        out.shape == x.shape
    ), f"Expected output shape {x.shape} but got {out.shape}"
    print(out.shape)
