import torch


def data_trans(ori_img_slice):
    """
    Spatial-spectral exchange
    :param ori_img_slice: tensor, input HSI img slice [N, c, slice_size, slice_size]
    :return: trans_img_slice: tensor, output HSI img slice [N, slice_size*slice_size, (c+2)/3, (c+2)/3]
    """
    N, c, slice_size, __ = ori_img_slice.shape
    slice_size_new = (c+2)//3
    trans_img_slice = torch.zeros(N, slice_size ** 2, slice_size_new, slice_size_new)
    ori_img_slice = ori_img_slice.reshape(N, c, -1)
    ori_img_slice = ori_img_slice.permute(0, 2, 1)

    for cc in range(slice_size_new):
        trans_img_slice[:, :, cc] = ori_img_slice[:, :, cc:cc + slice_size_new*2:2]

    return trans_img_slice

