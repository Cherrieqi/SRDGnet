import torch
import torch.nn as nn


def simi_cal(feature_1, feature_2):
    """
    :param feature_1:  [b, N, slice_size, slice_size]
    :param feature_2:  [b, N, slice_size, slice_size]
    :return:
    """
    b = feature_1.shape[0]
    feature_1 = torch.reshape(feature_1, [b, -1])
    feature_2 = torch.reshape(feature_2, [b, -1])

    norm_1 = torch.sqrt(torch.sum(torch.mul(feature_1, feature_1), dim=1))
    norm_2 = torch.sqrt(torch.sum(torch.mul(feature_2, feature_2), dim=1))

    feature_1 = feature_1/norm_1.unsqueeze(-1)
    feature_2 = feature_2/norm_2.unsqueeze(-1)

    similarity = feature_1@feature_2.t()

    return similarity


class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, similarity, cons_flag):
        """
        :param similarity:
        :param cons_flag: [batch_size] tensor 1/-1
        :return:
        """
        diff_vec = similarity - cons_flag*0.5 - 0.5
        diff_loss = torch.mean(diff_vec**2)

        return diff_loss


# loss 3 SDPloss
class SDPloss(nn.Module):
    def __init__(self):
        super(SDPloss, self).__init__()

    def forward(self, similarity_all, cons_flag, rate=0.2):
        """
        :param similarity_all: tuple ([batch_size, batch_size], [batch_size, batch_size]], ...)
        :param cons_flag: [batch_size] 1/-0.5
        :param rate: default=0.2
        :return:
        """
        pair_num = len(similarity_all)
        layer_weight = torch.tensor(range(pair_num))/pair_num

        prog_vec = 0
        for i in range(pair_num-1):
            prog_vec += (similarity_all[i+1]-similarity_all[i]-cons_flag*rate*layer_weight[i+1])**2

        # loss
        prog_loss = torch.mean(prog_vec)

        return prog_loss



