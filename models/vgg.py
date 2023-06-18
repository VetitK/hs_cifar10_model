"""
Create VGG-like network that has the following structure:
1. Stem:
    1. ConvNormAct 3 -> 32
2. Block 1
    1. ConvNormAct 32->32
    2. ConvNormAct 32->64
    3. MaxPool2d / 2
3. Block 2
    1. ConvNormAct 64->64
    2. ConvNormAct 64->128
    3. MaxPool2d / 2
4. Block 3
    1. ConvNormAct 128->128
    2. ConvNormAct 128->256
    3. MaxPool2d / 2
5. Block 4
    1. ConvNormAct 256 -> 256
    2. ConvNormAct 256 -> 256
6. Global AvgPool
7. Classifier 256->10
"""
import torch
from typing import Literal


class ConvNormAct(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(*args,
                            **kwargs),
            torch.nn.BatchNorm2d(args[1]),
            torch.nn.LeakyReLU()
        )

    def forward(self, input):
        return self.module.forward(input)


class VGG(torch.nn.Module):
    def __init__(self,
                 act=torch.nn.LeakyReLU):
        super().__init__()
        self.conv_modules = torch.nn.Sequential(
            ConvNormAct(3, 16, 3, padding=1), # stem
            ## block 1 3x32x32
            ConvNormAct(16, 16, 3, padding=1),
            ConvNormAct(16, 32, 3, padding=1),
            torch.nn.MaxPool2d(2),
            ## block 2 32x16x16
            ConvNormAct(32, 32, 3, padding=1),
            ConvNormAct(32, 64, 3, padding=1),
            torch.nn.MaxPool2d(2),
            ## block 3 64x8x8
            ConvNormAct(64, 64, 3, padding=1),
            ConvNormAct(64, 128, 3, padding=1),
            torch.nn.MaxPool2d(2),
            ## block 4 128x4x4
            ConvNormAct(128, 128, 3, padding=1),
            ConvNormAct(128, 256, 3, padding=1),
            torch.nn.MaxPool2d(2)
            # 256x2x2
        )
        # self.global_avg_pool = torch.nn.AdaptiveAvgPool2d([1, 1])
        self.classifier = torch.nn.Linear(256, 10)

    def forward(self, input):
        res = input # Nx3x32x32
        res = self.conv_modules(res)
        res = res.mean(dim=[-1, -2])
        res = self.classifier(res)
        return res