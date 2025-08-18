# SRDGnet

![SRDGnet](https://github.com/Cherrieqi/SRDGnet/blob/main/SRDGnet.png)


[A Shift Reduction Domain Generalization Network for Hyperspectral Image Cross-domain Classification](https://ieeexplore.ieee.org/document/11126530)




## Requirements

This code is based on **Python 3.10** and **Pytorch 1.12**.

*Installation list:*

**· pytorch**

**· matplotlib**

**· opencv-python**

**· scipy**

**· h5py**

**· tqdm**

**· scikit-learn**


## Models

**· SD--H13+H18 :** [model5.pth](https://pan.baidu.com/s/1Ha3yskj8WvRybBDWthvkIA?pwd=62vp)

**· SD--PU+PC :** [model5.pth](https://pan.baidu.com/s/1OUie4O6M_EWqFlwJjoC4nw?pwd=3ban)


## Datasets

**· [raw](https://pan.baidu.com/s/1UydVTlXiVtnpTHCzhjXGoA?pwd=wr7v) :** Houston2013 / Houston2018 / PaviaU / PaviaC

**· [H13+H18--PU/PC](https://pan.baidu.com/s/1bsbe7-zyiYrmDI2aV5hT-w?pwd=ttav) :** gen_H13 / gen_H18 / gen_PU / gen_PC

**· [PU+PC--H13/H18](https://pan.baidu.com/s/1BWqRwnT_0I4IzEpi7KwTfw?pwd=v3b8) :** gen_PU / gen_PC / gen_H13 / gen_H18 



## Getting start:

##### · Dataset structure

```
data/H1318
├── gen_H13
│   ├── SE_img.npy
│   ├── img.npy
│   └── gt.npy
├── gen_H18
│   ├── SE_img.npy
│   ├── img.npy
│   └── gt.npy
├── gen_PC
│   ├── SE_img.npy
│   ├── img.npy
│   └── gt.npy
└── gen_PU
    ├── SE_img.npy
    ├── img.npy
    └── gt.npy
```


```
data/PUPC
├── gen_PU
│   ├── SE_img.npy
│   ├── img.npy
│   └── gt.npy
├── gen_PC
│   ├── SE_img.npy
│   ├── img.npy
│   └── gt.npy
├── gen_H13
│   ├── SE_img.npy
│   ├── img.npy
│   └── gt.npy
└── gen_H18
    ├── SE_img.npy
    ├── img.npy
    └── gt.npy
```


```     
data/raw
├── Houston2013
│   ├── Houston.mat
│   └── Houston_gt.mat
├── Houston2018
│   ├── HoustonU.mat
│   └── HoustonU_gt.mat
├── PaviaC
│   ├── pavia.mat
│   └── pavia_gt.mat
└── PaviaU
     ├── paviaU.mat
     └── paviaU_gt.mat
```



**NOTE:**

​       Training and test data can be generated via *data_gen_xxxxx.py* respectively. Where *_H1318* indicates that the source domains are H13 and H18 and *_PUPC* indicates that the source domains are PU and PC.

##### · Train

​       Run *train_xxxx.py*. 

##### · Test

​       Run *test_xxxx.py*. 
