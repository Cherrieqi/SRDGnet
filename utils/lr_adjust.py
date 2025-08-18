import torch


def lr_adj(iter_num, change_num, lr_init, model, rate=0.2):
    lr = lr_init
    if iter_num < 0:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.1)
    else:
        if iter_num % change_num == 0 and iter_num != 0:
            lr = lr*rate
        else:
            lr = lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)

    return optimizer, lr


