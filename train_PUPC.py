import os
import time
import random
import warnings
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from networks.SRDGnet import SRDGnet
from torch.utils.data import DataLoader
from loss import simi_cal, SDPloss
from ImageDataset_PUPC import ImgDataset_train
from utils.lr_adjust import lr_adj
from sklearn.metrics import accuracy_score
from utils.draw_loss_curve import draw_loss_curve
from utils.cls_weight_calculation import weight_calc_HSI
from utils.ema import EMA

SE_img_1 = np.load("data/PUPC/gen_PC/SE_img.npy")
SE_img_1 = torch.from_numpy(SE_img_1).float()
img_1 = np.load("data/PUPC/gen_PC/img.npy")
img_1 = torch.from_numpy(img_1).float()
label_1 = np.load("data/PUPC/gen_PC/gt.npy")
label_1 = torch.LongTensor(label_1)

SE_img_2 = np.load("data/PUPC/gen_PU/SE_img.npy")
SE_img_2 = torch.from_numpy(SE_img_2).float()
img_2 = np.load("data/PUPC/gen_PU/img.npy")
img_2 = torch.from_numpy(img_2).float()
label_2 = np.load("data/PUPC/gen_PU/gt.npy")
label_2 = torch.LongTensor(label_2)


seed = 4
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

warnings.filterwarnings("ignore")

w = [3, 1, 1]
num_epoch = 5
batch_size = 1024
learning_rate = 0.02
rate = 0.15
change_num = 80
ema_epoch = 3

device = "cuda:0"
slice_size = 3
in_ch = slice_size**2
out_ch_shift = 128
out_ch_ifem = [128]*4
out_ch = [256, 64]
img_size = 35  # HSI SE_data size
band_num = 103
class_num = 4


work_dir = f'./work_dir/PUPC_inch{in_ch}_w{w[0]}-{w[1]}-{w[2]}_b{batch_size}_lr{learning_rate}/' \
           + time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

logs_path = work_dir + 'logs/'
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
with open(logs_path + "train_logs.txt", 'a') as f:
    f.write("Training loss logs:")
    f.write("\n")
f.close()

train_loss_list = []


train_set = ImgDataset_train(SE_img_1, img_1, label_1, SE_img_2, img_2, label_2)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


label_all = []
for i, data in enumerate(train_loader):
    for j in range(data[2].shape[0]):
        label_all.append(data[2][j].tolist())
        label_all.append(data[5][j].tolist())

label_all = torch.tensor(label_all).argmax(dim=1) + 1
label_all = label_all.numpy()
weight_cls = weight_calc_HSI(label_all, cls_id=list(range(1, class_num + 1)))
print(weight_cls)


model = SRDGnet(in_ch=in_ch, out_ch_shift=out_ch_shift, out_ch_ifem=out_ch_ifem, out_ch=out_ch,
                img_size=img_size, band_num=band_num, class_num=class_num, slice_size=slice_size).to(device)


loss_classify = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weight_cls.to(device))
loss_diff = nn.BCEWithLogitsLoss(reduction='mean')
loss_prog = SDPloss()

# start training
time_start = time.time()

for epoch in range(0, num_epoch):
    if epoch == ema_epoch:
        ema = EMA(model, decay=0.9)
        ema.register()
    f = open(logs_path + "train_logs.txt", 'a')
    epoch_start_time = time.time()
    train_loss = 0.0

    model.train()

    # set the tqmd
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    loop_len = len(loop)
    for i, data in loop:
        num_iters = epoch * loop_len + i
        optimizer, learning_rate = lr_adj(num_iters, change_num, learning_rate, model, rate)
        optimizer.zero_grad()

        y_1, side_1_1, side_1_2, side_1_3, side_1_4 = model(data[0].to(device), data[1].to(device))
        y_2, side_2_1, side_2_2, side_2_3, side_2_4 = model(data[3].to(device), data[4].to(device))

        simi_gt = torch.full([data[2].shape[0], data[2].shape[0]], 0.).to(device)
        for batch in range(data[2].shape[0]):
            cons = torch.where(torch.argmax(data[5], dim=1) == torch.argmax(data[2][batch]))
            simi_gt[batch, cons[0]] = 1

        # loss 1
        loss1_1 = loss_classify(y_1, data[2].float().to(device))
        loss1_2 = loss_classify(y_2, data[5].float().to(device))
        loss1 = 0.25*loss1_1 + 1.75*loss1_2

        # calculate the similarity between two SDs
        simi_1 = simi_cal(side_1_1, side_2_1)
        simi_2 = simi_cal(side_1_2, side_2_2)
        simi_3 = simi_cal(side_1_3, side_2_3)
        simi_4 = simi_cal(side_1_4, side_2_4)

        similarity_all = [simi_1, simi_2, simi_3, simi_4]

        # loss 2
        loss2_1 = loss_diff(simi_1, simi_gt)
        loss2_2 = loss_diff(simi_2, simi_gt)
        loss2_3 = loss_diff(simi_3, simi_gt)
        loss2_4 = loss_diff(simi_4, simi_gt)

        loss2 = loss2_1 + loss2_2 + loss2_3 + loss2_4

        # loss 3
        loss3 = loss_prog(similarity_all, simi_gt)

        batch_loss = w[0] * loss1 + w[1] * loss2 + w[2] * loss3  # 权重有待考虑

        batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss = train_loss + batch_loss.item()
            gt_1 = data[2].argmax(dim=1).flatten().cpu().numpy()
            pred_1 = y_1.argmax(dim=1).flatten().cpu().numpy()
            gt_2 = data[5].argmax(dim=1).flatten().cpu().numpy()
            pred_2 = y_2.argmax(dim=1).flatten().cpu().numpy()
            oa_1 = accuracy_score(gt_1, pred_1)
            oa_2 = accuracy_score(gt_2, pred_2)

        loop.set_description(f'Epoch [{epoch + 1}/{num_epoch}]')
        loop.set_postfix(cls_ls=loss1.item(), dif_ls=loss2.item(), prog_ls=loss3.item(),
                         b_ls=batch_loss.item(), lr=optimizer.state_dict()['param_groups'][0]['lr'],
                         oa_1=oa_1, oa_2=oa_2)
        optimizer.zero_grad()

    # ema
    if epoch >= ema_epoch:
        ema.apply_shadow()
    models_path = work_dir + 'models/'
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    torch.save(model.state_dict(), models_path + 'model{}.pth'.format(epoch + 1))

    print('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f' %
          (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss))

    f.write('[%03d/%03d] %2.2f sec(s) Train  Loss: %3.6f \n' %
            (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss))
    f.close()

    # training loss
    train_loss_list.append(train_loss)

time_end = time.time()
f.close()

# draw the loss curve
epoch_list = [(i + 1) for i in range(num_epoch)]
draw_loss_curve(epoch_list, train_loss=train_loss_list, save_path=logs_path + "loss.png")
print("training time:", time_end - time_start, 's')
f.close()
