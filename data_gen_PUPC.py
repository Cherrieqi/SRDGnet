import gc
import torch
import numpy as np

from utils.normHSI import normHSI_smp_s
from utils.interpolate import interpolate
from utils.data_split import data_split
from utils.data_trans import data_trans
from utils.readHSI import readHSI, label_trans, one_hot_slice
from utils.set_slc_division import set_division, set_division_pro, set_shuffle

rate_train_1 = [5000, 0.99999999, 3000, 0.999999999]
rate_train_2 = [0.99999999, 0.99999999, 6000, 4000]
rate_test = 0.999999999
slice_size = 3
class_num = 4

# readHSI --> normHSI_all --> label_trans --> data_split --> set_division --> spec_split --> data_trans
#
# Houston + PU
# PaviaU
path = './data/raw/PaviaU/'
image_name = 'paviaU'
label_name = 'paviaU_gt'
image_PU, label_PU = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])

PU_label = label_trans(label_PU, [2, 6, 4, 1], [1, 2, 3, 4])
del label_PU
gc.collect()

PU_image_slice, PU_label_slice, __ = data_split(slice_size, image_PU, PU_label)
del image_PU, PU_label
gc.collect()

PU_img_shuffle, PU_gt_shuffle, __ = set_shuffle(PU_image_slice, PU_label_slice)
del PU_image_slice, PU_label_slice
gc.collect()

PU_image, PU_gt = set_division_pro(4, [1, 2, 3, 4], PU_img_shuffle, PU_gt_shuffle, 'train', rate_train_1)
PU_image = normHSI_smp_s(PU_image)
del PU_img_shuffle, PU_gt_shuffle
gc.collect()

PU_gt_OH = one_hot_slice(PU_gt, class_num=class_num)
del PU_gt
gc.collect()

np.save(f"data/PUPC/gen_PU/gt.npy", PU_gt_OH)
del PU_gt_OH
gc.collect()

np.save(f"data/PUPC/gen_PU/img.npy", PU_image)

PU_SE_image = data_trans(PU_image)
del PU_image
gc.collect()

np.save(f"data/PUPC/gen_PU/SE_img.npy", PU_SE_image)
del PU_SE_image
gc.collect()


# PaviaC
path = './data/raw/PaviaC/'
image_name = 'pavia'
label_name = 'pavia_gt'
image_PC, label_PC = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])

image_PC_new = torch.zeros([103, image_PC.shape[1], image_PC.shape[2]])
image_PC_new[0] = image_PC[0]
image_PC_new[1:] = image_PC
del image_PC
gc.collect()

PC_label = label_trans(label_PC, [3, 5, 2, 6], [1, 2, 3, 4])
del label_PC
gc.collect()

PC_image_slice, PC_label_slice, __ = data_split(slice_size, image_PC_new, PC_label)
del image_PC_new, PC_label
gc.collect()

PC_img_shuffle, PC_gt_shuffle, __ = set_shuffle(PC_image_slice, PC_label_slice)
del PC_image_slice, PC_label_slice
gc.collect()

PC_image, PC_gt = set_division_pro(4, [1, 2, 3, 4], PC_img_shuffle, PC_gt_shuffle, 'train', rate_train_2)
PC_image = normHSI_smp_s(PC_image)
del PC_img_shuffle, PC_gt_shuffle
gc.collect()

PC_gt_OH = one_hot_slice(PC_gt, class_num=class_num)
del PC_gt
gc.collect()

np.save(f"data/PUPC/gen_PC/gt.npy", PC_gt_OH)
del PC_gt_OH
gc.collect()

np.save(f"data/PUPC/gen_PC/img.npy", PC_image)

PC_SE_image = data_trans(PC_image)
del PC_image
gc.collect()

np.save(f"data/PUPC/gen_PC/SE_img.npy", PC_SE_image)
del PC_SE_image
gc.collect()


# Houston2013
path = './data/raw/Houston2013/'
image_name = 'Houston'
label_name = 'Houston_gt'
image_H13, label_H13 = readHSI(path, image_name, label_name, mode=0, img_order=[2, 0, 1])

image_H13 = image_H13[4:103+4]

H13_label = label_trans(label_H13, [1, 5, 4, 9], [1, 2, 3, 4])
del label_H13
gc.collect()

H13_image_slice, H13_label_slice, H13_row_col = data_split(slice_size, image_H13, H13_label)
del image_H13, H13_label
gc.collect()

H13_image, H13_gt, H13_point_idx = set_division(4, [1, 2, 3, 4], H13_image_slice, H13_label_slice, 'train', rate_test, H13_row_col)
H13_image = normHSI_smp_s(H13_image)
del H13_image_slice, H13_label_slice
gc.collect()

H13_gt_OH = one_hot_slice(H13_gt, class_num=class_num)
del H13_gt
gc.collect()

np.save(f"data/PUPC/gen_H13/gt.npy", torch.cat((H13_gt_OH, H13_point_idx), dim=1))
del H13_gt_OH
gc.collect()

np.save(f"data/PUPC/gen_H13/img.npy", H13_image)

H13_SE_image = data_trans(H13_image)
del H13_image
gc.collect()

np.save(f"data/PUPC/gen_H13/SE_img.npy", H13_SE_image)
del H13_SE_image
gc.collect()


# Houston2018
path = './data/raw/Houston2018/'
image_name = 'HoustonU'
label_name = 'HoustonU_gt'
image_H18_raw, label_H18 = readHSI(path, image_name, label_name, mode=1, img_order=[0, 1, 2])
image_H18 = interpolate(image_H18_raw[:48], 144)
image_H18 = image_H18[4:4+103]
del image_H18_raw
gc.collect()

H18_label = label_trans(label_H18, [1, 6, 4, 10], [1, 2, 3, 4])
del label_H18
gc.collect()

H18_image_slice, H18_label_slice, H18_row_col = data_split(slice_size, image_H18, H18_label)
del image_H18, H18_label
gc.collect()

H18_image, H18_gt, H18_point_idx = set_division(4, [1, 2, 3, 4], H18_image_slice, H18_label_slice, 'train', rate_test, H18_row_col)
H18_image = normHSI_smp_s(H18_image)
del H18_image_slice, H18_label_slice
gc.collect()

H18_gt_OH = one_hot_slice(H18_gt, class_num=class_num)
del H18_gt
gc.collect()

np.save(f"data/PUPC/gen_H18/gt.npy", torch.cat((H18_gt_OH, H18_point_idx), dim=1))
del H18_gt_OH
gc.collect()

np.save(f"data/PUPC/gen_H18/img.npy", H18_image)

H18_SE_image = data_trans(H18_image)
del H18_image
gc.collect()

np.save(f"data/PUPC/gen_H18/SE_img.npy", H18_SE_image)
del H18_SE_image
gc.collect()

