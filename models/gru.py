import torch

class GRUConvX(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module = torch.nn.GRU(in_channels, 
                                   out_channels, 
                                   bidirectional=True)

    def forward(self, input):
        batch_size = input.shape[0]
        # print(input.shape, '<- input shape')
        IMG = input
        IMG = IMG.permute([2, 1, 0, 3])
        IMG = IMG.reshape([IMG.shape[0], IMG.shape[1], -1])
        IMG = IMG.permute([0, 2, 1])
        # print(IMG.shape, '<- input to GRU')
        RES, MEM = self.module(IMG) #SIGNAL #BATCH #CHANNELS
        # print(RES.shape, '<- output from GRU')
        RES = RES.permute([0, 2, 1])
        # print(RES.shape, '<- before reshape')
        RES = RES.reshape([RES.shape[0], RES.shape[1], batch_size, -1]) #LENGTH #CHAN #BATCH #DIM
        RES = RES.permute([2, 1, 0, 3])
        return RES


class GRUConvY(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module = torch.nn.GRU(in_channels, 
                                   out_channels, 
                                   bidirectional=True)

    def forward(self, input):
        batch_size = input.shape[0]

        IMG = input
        IMG = IMG.permute([3, 1, 0, 2])
        IMG = IMG.reshape([IMG.shape[0], IMG.shape[1], -1])
        IMG = IMG.permute([0, 2, 1])
        RES, MEM = self.module(IMG) #SIGNAL #BATCH #CHANNELS
        # RES = IMG
        RES = RES.permute([0, 2, 1])
        RES = RES.reshape([RES.shape[0], RES.shape[1], batch_size, -1]) #LENGTH #CHAN #BATCH #DIM
        RES = RES.permute([2, 1, 3, 0])
        return RES

class GRUConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, m_channels):
        super().__init__()
        self.GRU_X = GRUConvX(in_channels, m_channels) # x2
        self.GRU_Y = GRUConvY(in_channels, m_channels) # x2
        self.out_conv = torch.nn.Conv2d(4 * m_channels, out_channels, 1)

    def forward(self, input):
        res = input
        x_res = self.GRU_X.forward(res)
        y_res = self.GRU_Y.forward(res)
        res = torch.cat([x_res, y_res], dim=1) # problem
        res = self.out_conv(res)
        return res


class GRU_VGG(torch.nn.Module):
    def __init__(self,
                 act=torch.nn.LeakyReLU):
        super().__init__()
        self.conv_modules = torch.nn.Sequential(
            GRUConv(3, 16, 16), # stem
            ## block 1 3x32x32
            GRUConv(16, 16, 16),
            GRUConv(16, 32, 32),
            torch.nn.MaxPool2d(2),
            # ## block 2 32x16x16
            GRUConv(32, 32, 32),
            GRUConv(32, 64, 64),
            torch.nn.MaxPool2d(2),
            # ## block 3 64x8x8
            GRUConv(64, 64, 64),
            GRUConv(64, 128, 128),
            torch.nn.MaxPool2d(2),
            # ## block 4 128x4x4
            GRUConv(128, 128, 128),
            GRUConv(128, 256, 256),
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
