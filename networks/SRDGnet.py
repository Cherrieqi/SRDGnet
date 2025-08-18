import torch
import torch.nn as nn
from networks.SE_AEM import SE_AEM
from networks.cal_domain_shift import cal_domain_shift
from networks.IFEH import IFEH
from networks.IFEM import IFEM


class SRDGnet(nn.Module):
    def __init__(self, in_ch, out_ch_shift, out_ch_ifem: list, out_ch: list, img_size, band_num, class_num, slice_size):
        super(SRDGnet, self).__init__()
        self.se_aem = SE_AEM(in_ch, out_ch_shift, img_size)
        self.cal_domain_shift = cal_domain_shift(out_ch_shift, slice_size)
        self.ifeh = IFEH(band_num, out_ch_shift)
        self.ifem = IFEM(out_ch_shift, out_ch_ifem, class_num)

        self.classifier = nn.Sequential(nn.Conv2d(out_ch_ifem[3], out_ch[0], kernel_size=slice_size, padding=0, groups=class_num),
                                        nn.Conv2d(out_ch[0], out_ch[1], kernel_size=1, padding=0, groups=class_num),
                                        nn.Conv2d(out_ch[1], class_num, kernel_size=1, padding=0))

    def forward(self, x_SE, x_ori):
        x_SE = torch.tanh(x_SE)
        x_ori = torch.tanh(x_ori)
        # SE-AEM
        y_local = self.se_aem(x_SE)
        # SE-IFEH
        y_global = self.ifeh(x_ori)
        # cal_domain_shift
        y_shift = self.cal_domain_shift(y_local, y_global)
        # SE-IFEM
        y, y1, y2, y3, y4 = self.ifem(y_shift)

        # classifier
        y = self.classifier(y)
        y = y.squeeze(-1).squeeze(-1)
        # print(y)

        return y, y1, y2, y3, y4

