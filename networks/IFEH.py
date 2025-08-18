import torch.nn as nn
from networks.SE_AEM import CBR


class IFEH(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(IFEH, self).__init__()
        self.ifeh = nn.Sequential(CBR(in_ch, out_ch),
                                  CBR(out_ch, out_ch, kernel_size=1, padding=0),
                                  CBR(out_ch, out_ch, kernel_size=1, padding=0))

    def forward(self, x):
        y = self.ifeh(x)

        return y

