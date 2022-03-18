import torch
import torch.nn as nn

from models.modules import ResidualBlock, UpSample


class Generator(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 3, n_residual: int = 16):
        super(Generator, self).__init__()

        # First Conv Layer
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
                                        nn.PReLU())
        # residual layers
        residual_blocks = []
        for _ in range(n_residual):
            residual_blocks.append(ResidualBlock(channels=64))
        self.residual_blocks = nn.Sequential(*residual_blocks)
        # Second Conv Layer
        self.second_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(64, 0.8))
        # UpSampling Layer
        self.first_up = UpSample(in_channels=64, out_channels=256, kernel_size=3,
                                 stride=1, padding=1, upscale_factor=2)
        self.second_up = UpSample(in_channels=64, out_channels=256, kernel_size=3,
                                  stride=1, padding=1, upscale_factor=2)
        # Last Conv Layer
        self.last_conv = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4),
                                       nn.Tanh())

    def forward(self, inputs: torch.Tensor):
        first_conv = self.first_conv(inputs)
        residual_blocks = self.residual_blocks(first_conv)
        second_conv = self.second_conv(residual_blocks)
        out = torch.add(first_conv, second_conv)

        up1 = self.first_up(out)
        up2 = self.second_up(up1)
        out = self.last_conv(up2)
        return out


if __name__ == "__main__":
    t = torch.randn(2, 3, 256//4, 256//4).cuda()
    g = Generator().cuda()
    print(g(t).size()) # [2, 3, 256, 256]
