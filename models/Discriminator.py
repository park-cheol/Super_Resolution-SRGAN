# 원래 논분에서는 마지막 layer에 2개의 Linear사용하여 구함
# 여기서는 PatchGAN Discriminator(LSGAN Loss도)를 이용
import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.conv_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.conv_3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.conv_4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.conv_5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.conv_6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.conv_7 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.conv_8 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2, inplace=True))

        self.last_conv = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs: torch.Tensor):
        out = self.conv_1(inputs)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = self.conv_6(out)
        out = self.conv_7(out)
        out = self.conv_8(out)
        out = self.last_conv(out)

        return out


if __name__ == "__main__":
    t = torch.randn(2, 3, 256, 256).cuda()
    d = Discriminator().cuda()

    print(d(t).size()) # [2, 1, 16, 16]
