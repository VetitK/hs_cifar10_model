"""
1. Stem:
    1. ConvNormAct 3 -> 32
2. Block 1
    1. ResBlock(ConvNormAct 32->32/4->32) x 4
    2. ResBlock(ConvNormAct 32->64, stride=2)
3. Block 2
    1. ResBlock(ConvNormAct 64->64/4->64) x 4
    2. ResBlock(ConvNormAct 64->128, stride=2)
4. Block 3
    1. ResBlock(ConvNormAct 128->128/4->128) x 4
    2. ResBlock(ConvNormAct 128->256, stride=2)
5. Block 4
    1. ResBlock(ConvNormAct 256->256/4->256) x 5
6. Global AvgPool
7. Classifier 256->10
"""
import torch
from models.vgg import ConvNormAct
class ResBlock(torch.nn.Module):
    def __init__(self, module, bypass=torch.nn.Identity()):
        super().__init__()
        self.module = module
        self.bypass = bypass

    def forward(self, input):
        return self.module(input) + self.bypass(input)


class ResNetDW(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = torch.nn.Sequential(
            ConvNormAct(3, 16, 3, padding=1), # stem

            # block 1 32x32
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(16, 32, 1),
                    ConvNormAct(32, 32, 5, padding=2, groups=32),
                    ConvNormAct(32, 16, 1),
                    ConvNormAct(16, 32, 1),
                    ConvNormAct(32, 32, 5, padding=2, groups=32),
                    ConvNormAct(32, 16, 1),
                    )),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(16, 32, 1),
                    ConvNormAct(32, 32, 5, padding=2, groups=32),
                    ConvNormAct(32, 16, 1),
                    ConvNormAct(16, 32, 1),
                    ConvNormAct(32, 32, 5, padding=2, groups=32),
                    ConvNormAct(32, 16, 1),
                    )),            
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(16, 32, 1),
                    ConvNormAct(32, 32, 5, padding=2, groups=32),
                    ConvNormAct(32, 16, 1),
                    ConvNormAct(16, 32, 1),
                    ConvNormAct(32, 32, 5, padding=2, groups=32),
                    ConvNormAct(32, 16, 1),
                    )),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(16, 32, 1),
                    ConvNormAct(32, 32, 5, padding=2, groups=32),
                    ConvNormAct(32, 16, 1),
                    ConvNormAct(16, 32, 1),
                    ConvNormAct(32, 32, 5, padding=2, groups=32),
                    ConvNormAct(32, 32, 1)),
                    bypass=torch.nn.Conv2d(16, 32, 1, stride=2)),
            
            # block 2
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(32, 8, 3, padding=1),
                    ConvNormAct(8, 32, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(32, 8, 3, padding=1),
                    ConvNormAct(8, 32, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(32, 8, 3, padding=1),
                    ConvNormAct(8, 32, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(32, 8, 3, padding=1, stride=2),
                    ConvNormAct(8, 64, 3, padding=1)),
                    bypass=torch.nn.Conv2d(32, 64, 1, stride=2)),
            # block 3
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(64, 16, 3, padding=1),
                    ConvNormAct(16, 64, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(64, 16, 3, padding=1),
                    ConvNormAct(16, 64, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(64, 16, 3, padding=1),
                    ConvNormAct(16, 64, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(64, 32, 3, padding=1, stride=2),
                    ConvNormAct(32, 128, 3, padding=1)),
                    bypass=torch.nn.Conv2d(64, 128, 1, stride=2)),
            # block 4
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(128, 32, 3, padding=1),
                    ConvNormAct(32, 128, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(128, 32, 3, padding=1),
                    ConvNormAct(32, 128, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(128, 32, 3, padding=1),
                    ConvNormAct(32, 128, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(128, 64, 3, padding=1, stride=2),
                    ConvNormAct(64, 256, 3, padding=1)),
                    bypass=torch.nn.Conv2d(128, 256, 1, stride=2)),
        )
        self.classifier = torch.nn.Linear(256, 10)

    def forward(self, input):
        res = input
        res = self.conv_net(res)
        res = res.mean(dim=[-1, -2])
        res = self.classifier(res)
        return res


class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = torch.nn.Sequential(
            ConvNormAct(3, 16, 3, padding=1), # stem

            # block 1 32x32
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(16, 4, 3, padding=1),
                    ConvNormAct(4, 16, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(16, 4, 3, padding=1),
                    ConvNormAct(4, 16, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(16, 4, 3, padding=1),
                    ConvNormAct(4, 16, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(16, 8, 3, stride=2, padding=1),
                    ConvNormAct(8, 32, 3, padding=1)),
                    bypass=torch.nn.Conv2d(16, 32, 1, stride=2)),
            
            # block 2
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(32, 8, 3, padding=1),
                    ConvNormAct(8, 32, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(32, 8, 3, padding=1),
                    ConvNormAct(8, 32, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(32, 8, 3, padding=1),
                    ConvNormAct(8, 32, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(32, 8, 3, padding=1, stride=2),
                    ConvNormAct(8, 64, 3, padding=1)),
                    bypass=torch.nn.Conv2d(32, 64, 1, stride=2)),
            # block 3
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(64, 16, 3, padding=1),
                    ConvNormAct(16, 64, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(64, 16, 3, padding=1),
                    ConvNormAct(16, 64, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(64, 16, 3, padding=1),
                    ConvNormAct(16, 64, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(64, 32, 3, padding=1, stride=2),
                    ConvNormAct(32, 128, 3, padding=1)),
                    bypass=torch.nn.Conv2d(64, 128, 1, stride=2)),
            # block 4
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(128, 32, 3, padding=1),
                    ConvNormAct(32, 128, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(128, 32, 3, padding=1),
                    ConvNormAct(32, 128, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(128, 32, 3, padding=1),
                    ConvNormAct(32, 128, 3, padding=1))),
            ResBlock(
                torch.nn.Sequential(
                    ConvNormAct(128, 64, 3, padding=1, stride=2),
                    ConvNormAct(64, 256, 3, padding=1)),
                    bypass=torch.nn.Conv2d(128, 256, 1, stride=2)),
        )
        self.classifier = torch.nn.Linear(256, 10)

    def forward(self, input):
        res = input
        res = self.conv_net(res)
        res = res.mean(dim=[-1, -2])
        res = self.classifier(res)
        return res