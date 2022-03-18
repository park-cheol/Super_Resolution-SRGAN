import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:18])

    def forward(self, inputs: torch.Tensor):
        out = self.feature_extractor(inputs)

        return out
