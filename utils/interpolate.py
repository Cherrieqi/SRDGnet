import scipy as sp
import numpy as np
import torch


def interpolate(image, c_new):
    """
    :param image: tensor [c, h, w]
    :param c_new: int
    :return: image_new: [c_new, h, w]
    """
    c, h, w = image.shape[0], image.shape[1], image.shape[2]
    image = np.array(image)
    image_new = np.zeros([c_new, h, w])
    for i in range(h):
        for j in range(w):
            x = np.linspace(0, c-1, num=c)
            y = image[:, i, j]
            # f1 = sp.interpolate.interp1d(x, y, kind='cubic')
            f1 = sp.interpolate.interp1d(x, y, kind='linear')
            x_new = np.linspace(0, c-1, num=c_new)
            image_new[:, i, j] = f1(x_new)
    image_new = torch.FloatTensor(image_new)

    return image_new


