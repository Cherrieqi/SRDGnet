import torch


def normHSI_smp_s(image, rate=1, eps=0.00000000001):
    image_norm = torch.zeros(image.shape)
    for i in range(image.shape[0]):
        max_value = torch.max(image[i])
        min_value = torch.min(image[i])
        image_norm[i] = rate * (image[i] - min_value) / (max_value - min_value+eps)

    return image_norm


