import torch.nn as nn
from networks.SE_AEM import CBR


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, class_num=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False, groups=class_num)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False, groups=class_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, cls_num=1, kernel_size=1):
        super(conv_block, self).__init__()
        self.net_main = nn.Sequential(CBR(in_ch, out_ch, kernel_size=kernel_size, padding=0, groups=cls_num),
                                      CBR(out_ch, 2 * out_ch, kernel_size=1, padding=0, groups=cls_num),
                                      CBR(2 * out_ch, out_ch, kernel_size=1, padding=0))
        self.ca = ChannelAttention(out_ch, class_num=cls_num)
        self.conv = CBR(in_ch, out_ch, kernel_size=kernel_size, padding=0, groups=cls_num)

    def forward(self, x_main):
        y_main = self.net_main(x_main)
        y = y_main + self.conv(x_main)
        w = self.ca(y)
        y = y*w

        return y


class IFEM(nn.Module):
    def __init__(self, in_ch, out_ch: list, cls_num):
        super(IFEM, self).__init__()
        self.block1 = conv_block(in_ch, out_ch[0], cls_num)
        self.block2 = conv_block(out_ch[0], out_ch[1], cls_num)
        self.block3 = conv_block(out_ch[1], out_ch[2], cls_num)
        self.block4 = conv_block(out_ch[2], out_ch[3], cls_num)

    def forward(self, x_shift):
        y_main_1 = self.block1(x_shift)
        y_main_2 = self.block2(y_main_1)
        y_main_3 = self.block3(y_main_2)
        y_main_4 = self.block4(y_main_3)

        return y_main_4, y_main_1, y_main_2, y_main_3, y_main_4
