"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import torch.nn as nn
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
# from models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info


def train(version='mini',
          dataroot='/home/cuidongdong/nuscenes_mini',
          nepochs=10000,
          gpuid=1,
          H=900,  # 图片大小
          W=1600,
          resize_lim=(0.193, 0.225),    # resize的范围
          final_dim=(128, 352),         # 数据预处理之后图片最后的尺寸
          bot_pct_lim=(0.0, 0.22),      # 裁剪图片时，图像底部裁减掉部分所占的比例
          rot_lim=(-5.4, 5.4),          # 训练时旋转图片的角度范围
          rand_flip=True,               # 随机翻转
          ncams=5,                      # 摄像机通道数
          max_grad_norm=5.0,
          pos_weight=2.13,              # 损失函数中给正样本项损失乘的权重系数
          logdir='./runs',              # 日志的输出文件

          xbound=[-50.0, 50.0, 0.5],    # 限制x的方向范围并进行划分网格
          ybound=[-50.0, 50.0, 0.5],    # 限制y的方向范围并进行划分网格
          zbound=[-10.0, 10.0, 20.0],   # 限制z的方向范围并进行划分网格
          dbound=[4.0, 45.0, 1.0],      # 限制深度方向范围并进行划分网格

          bsz=4,                        # batch_size
          nworkers=10,                  # 线程数
          lr=1e-3,                      # 学习率
          weight_decay=1e-7,            # 权重衰减系数
          ):
    grid_conf = {                       # 网格配置
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {                   # 数据增强配置
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 1000 if version == 'mini' else 10000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        # np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                          rots.to(device),
                          trans.to(device),
                          intrins.to(device),
                          post_rots.to(device),
                          post_trans.to(device),
                          )
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
