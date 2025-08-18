import os
import random
import warnings
import torch
import numpy as np
from tqdm import tqdm

from networks.SRDGnet import SRDGnet
from torch.utils.data import DataLoader
from ImageDataset_H1318 import ImgDataset_test
from utils.label_vision import label_vision_1d
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

seed = 4
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

warnings.filterwarnings("ignore")

work_dir = f"./work_dir/H1318_inch9_w3-1-1_b1024_lr0.02/{seed}/"
for j in range(2):
    print(f"seed: {seed}")
    if j == 0:
        data_name = "PU"
        print("PU: \n")
    else:
        data_name = "PC"
        print("PC: \n")

    batch_size = 4096
    slice_size = 3
    in_ch = slice_size ** 2
    out_ch_shift = 128
    out_ch_ifem = [128] * 4
    out_ch = [256, 64]
    img_size = 35  # HSI SE_data size
    band_num = 103
    class_num = 4
    device = "cuda:0"

    model_path = work_dir + "models/model5.pth"
    result_save_path = work_dir + "results/"
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    if data_name == "PU":
        SE_img = np.load("data/H1318/gen_PU/SE_img.npy")
        img = np.load("data/H1318/gen_PU/img.npy")
        label = np.load("data/H1318/gen_PU/gt.npy")
        SE_img = torch.from_numpy(SE_img).float()
        img = torch.from_numpy(img).float()
        label = torch.LongTensor(label)
    else:
        SE_img = np.load("data/H1318/gen_PC/SE_img.npy")
        img = np.load("data/H1318/gen_PC/img.npy")
        label = np.load("data/H1318/gen_PC/gt.npy")
        SE_img = torch.from_numpy(SE_img).float()
        img = torch.from_numpy(img).float()
        label = torch.LongTensor(label)


    test_set = ImgDataset_test(SE_img, img, label)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = SRDGnet(in_ch=in_ch, out_ch_shift=out_ch_shift, out_ch_ifem=out_ch_ifem, out_ch=out_ch,
                    img_size=img_size, band_num=band_num, class_num=class_num, slice_size=slice_size).to(device)
    model.load_state_dict(torch.load(model_path))

    # test
    correct_num = 0
    gt_total = []
    pred_total = []
    row_col_total = []
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, data in loop:
            y, __, __, __, __ = model(data[0].to(device), data[1].to(device))
            gt = data[2][:, :4].argmax(dim=1).flatten().cpu().numpy()
            row_col = data[2][:, 4:]
            pred_prob = torch.sigmoid(y)
            pred = pred_prob.argmax(dim=1).flatten().cpu().numpy()
            gt_total.extend(gt)
            pred_total.extend(pred)
            row_col_total.extend(row_col)
            oa_batch = np.sum(gt - pred == 0) / data[0].shape[0]

            loop.set_description(f'[{i}/{len(test_loader)}]')
            loop.set_postfix(oa_batch=oa_batch)

    c_m = confusion_matrix(gt_total, pred_total)
    acc = accuracy_score(gt_total, pred_total)
    kappa = cohen_kappa_score(gt_total, pred_total)
    per_cls_acc_list = []
    for i in range(class_num):
        per_cls_acc_list.append(round(c_m[i][i]/sum(c_m[i])*100, 2))

    print("confusion matrix: \n", c_m)
    print("per class accuracy: \n", *per_cls_acc_list)
    print("accuracy: \n", round(acc*100, 2))
    print("kappa: \n", round(kappa*100, 2))

    with open(result_save_path + str(data_name) + "results.txt", 'a') as f:
        f.write("confusion matrix: \n")
        f.write(str(c_m))
        f.write("\n")
        f.write("per class accuracy: \n")
        f.write(str(per_cls_acc_list))
        f.write("\n")
        f.write("accuracy: \n")
        f.write(str(round(acc*100, 2)))
        f.write("\n")
        f.write("kappa: \n")
        f.write(str(round(kappa*100, 2)))
        f.write("\n")
        f.write("\n")
        f.close()

    if data_name == "PU":
        label_vision_1d(pred_total, row_col_total, 610, 340, result_save_path + "pred_H1318_PU.png")
    else:
        label_vision_1d(pred_total, row_col_total, 1096, 715, result_save_path + "pred_H1318_PC.png")
