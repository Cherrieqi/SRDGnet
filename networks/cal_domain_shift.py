import torch
import torch.nn as nn


class cal_domain_shift(nn.Module):
    def __init__(self, in_ch, slice_size):
        super(cal_domain_shift, self).__init__()
        self.weight_1 = nn.Parameter(torch.ones(in_ch, slice_size, slice_size), requires_grad=True)
        self.weight_2 = nn.Parameter(torch.ones(in_ch, slice_size, slice_size), requires_grad=True)

    def forward(self, x, x_global):

        x_shift = self.weight_1 * x + self.weight_2 * x_global

        return x_shift

