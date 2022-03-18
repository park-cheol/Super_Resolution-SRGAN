import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()

        self.blocks = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(channels, 0.8),
                                    nn.PReLU(),
                                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(channels, 0.8))

    def forward(self, inputs: torch.Tensor):
        out = self.blocks(inputs)
        out += inputs
        return out


class UpSample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, padding: int, upscale_factor:int):
        super(UpSample, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pix_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.act = nn.PReLU()

    def forward(self, inputs: torch.Tensor):
        out = self.conv(inputs)
        out = self.bn(out)
        out = self.pix_shuffle(out)
        out = self.act(out)

        return out

