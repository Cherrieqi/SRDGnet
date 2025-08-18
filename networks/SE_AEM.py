import torch.nn as nn
import math
from networks.CBAM import SpatialGate


def CBR(in_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class DC_block(nn.Module):
    def __init__(self, in_ch, kernel_size, padding, dilation=1, stride=1):
        super(DC_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, dilation=dilation),
                                  nn.BatchNorm2d(in_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.conv(x)

        return y


# SE-AEM
class SE_AEM(nn.Module):
    def __init__(self, in_ch, out_ch, img_size):
        super(SE_AEM, self).__init__()
        self.conv1_1 = nn.Sequential(DC_block(in_ch, kernel_size=5, padding=2, stride=3),
                                     SpatialGate(),
                                     nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
                                     DC_block(in_ch, kernel_size=5, padding=2, stride=1),
                                     SpatialGate(),
                                     nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
                                     DC_block(in_ch, kernel_size=5, padding=2, stride=1),
                                     SpatialGate(),
                                     nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
                                     )

        out_size = math.floor((img_size-1)/3)+1
        self.conv2 = nn.Conv2d(out_size*out_size, out_ch, kernel_size=1, padding=0)
        self.out_size = out_size
        self.in_ch = in_ch

    def forward(self, x):
        y1 = self.conv1_1(x)
        y = y1.reshape(y1.shape[0], self.in_ch, int(self.out_size**2))
        y = y.permute(0, 2, 1)
        y = y.reshape(y.shape[0], y.shape[1], int(pow(self.in_ch, 0.5)), int(pow(self.in_ch, 0.5)))
        y = self.conv2(y)

        return y

