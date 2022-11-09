import argparse
import os
import pdb
import torch
import time
import sys
sys.path.append("..")
import random
import torch.nn as nn
import numpy as np
import datetime
from tqdm import tqdm
import torch.distributed as dist
import torch.utils.data.distributed
from common_tools.logger import Logger
from loss import Loss
from timm.utils import AverageMeter
import torch.multiprocessing as mp
from utils import setup_seed
from model import PointPillars
from dataset import Kitti, get_dataloader
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration Parameters')

    parser.add_argument('--data_root', default='/data/cdd_data/kitti/detection',
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='outputs')

    parser.add_argument('--batch_size', type=int, default=96)

    parser.add_argument('--num_workers', type=int, default=6)

    parser.add_argument('--nclasses', type=int, default=3)

    parser.add_argument('--init_lr', type=float, default=0.00025)

    parser.add_argument('--max_epoch', type=int, default=25)

    parser.add_argument('--log_freq', type=int, default=8)

    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3467', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=6, type=int,
                        help='GPU id to use. 单卡才会使用')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=[2, 3, 4, 5, 6, 7], help="gpu设备编号")

    parser.add_argument('--multiprocessing_distributed', default=True)

    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    args = parser.parse_args()
    return args


args = parse_arguments()
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
os.makedirs(args.saved_path, exist_ok=True)
log = Logger(os.path.join(args.saved_path, 'pointpillars.log'), level='info')


def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)


# 写入writer
saved_logs_path = os.path.join(args.saved_path, 'summary')
os.makedirs(saved_logs_path, exist_ok=True)
writer = SummaryWriter(saved_logs_path)
saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
os.makedirs(saved_ckpt_path, exist_ok=True)


def main():
    setup_seed(0)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        # ngpus_per_node = torch.cuda.device_count()
        ngpus_per_node = len(args.gpu_devices)

    else:
        ngpus_per_node = 1

    print("the ngpus_per_node is: ", ngpus_per_node)
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = PointPillars(nclasses=args.nclasses)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if torch.cuda.is_available():
            if args.gpu is not None:
                print("args.distributed: true", args.gpu)
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataset = Kitti(data_root=args.data_root,
                          split='train')
    val_dataset = Kitti(data_root=args.data_root,
                        split='val')
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader(dataset=val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False)

    loss_func = Loss().to(device)

    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=init_lr,
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=init_lr * 10,
                                                    total_steps=max_iters,
                                                    pct_start=0.4,
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True,
                                                    base_momentum=0.95 * 0.895,
                                                    max_momentum=0.95,
                                                    div_factor=10)


    num_steps = len(train_dataloader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    for epoch in range(args.max_epoch):
        # log.logger.info('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0
        optimizer.zero_grad()
        for i, data_dict in enumerate(train_dataloader):
            end = time.time()

            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()

            batched_pts = data_dict['batched_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
            batched_difficulty = data_dict['batched_difficulty']
            bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                model(batched_pts=batched_pts,
                             mode='train',
                             batched_gt_bboxes=batched_gt_bboxes,
                             batched_gt_labels=batched_labels)

            bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
            bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

            batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
            batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
            batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
            # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
            batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
            # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)

            pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
            bbox_pred = bbox_pred[pos_idx]
            batched_bbox_reg = batched_bbox_reg[pos_idx]
            # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
            bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
            batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
            bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
            batched_dir_labels = batched_dir_labels[pos_idx]

            num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
            bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
            batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
            batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

            loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                  bbox_pred=bbox_pred,
                                  bbox_dir_cls_pred=bbox_dir_cls_pred,
                                  batched_labels=batched_bbox_labels,
                                  num_cls_pos=num_cls_pos,
                                  batched_bbox_reg=batched_bbox_reg,
                                  batched_dir_labels=batched_dir_labels)

            loss = loss_dict['total_loss']
            loss.backward()

            torch.cuda.synchronize()
            loss_meter.update(loss.item(), 16)
            batch_time.update(time.time() - end)
            # torch.nn.utils.clip_grad_norm_(pointpillars.parameters(), max_norm=35)
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1
            if dist.get_rank() == 0:
                log.logger.info(
                    f'Train: [{epoch}/{args.max_epoch}][{i}/{len(train_dataloader)}]\t'
                    # f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    # f'mem {memory_used:.0f}MB'
                )
            if global_step % args.log_freq == 0 and dist.get_rank() == 0:
                save_summary(writer, loss_dict, global_step, 'train',
                             lr=optimizer.param_groups[0]['lr'],
                             momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1
        if (epoch + 1) % args.ckpt_freq_epoch == 0 and dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(saved_ckpt_path, f'epoch_{epoch + 1}.pth'))

        if epoch % 2 == 0:
            continue
        model.eval()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                if not args.no_cuda:
                    # move the tensors to the cuda
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()

                batched_pts = data_dict['batched_pts']
                batched_gt_bboxes = data_dict['batched_gt_bboxes']
                batched_labels = data_dict['batched_labels']
                batched_difficulty = data_dict['batched_difficulty']
                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = \
                    model(batched_pts=batched_pts,
                                 mode='train',
                                 batched_gt_bboxes=batched_gt_bboxes,
                                 batched_gt_labels=batched_labels)

                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                # batched_bbox_reg_weights = anchor_target_dict['batched_bbox_reg_weights'].reshape(-1)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
                # batched_dir_labels_weights = anchor_target_dict['batched_dir_labels_weights'].reshape(-1)

                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1]) * torch.cos(batched_bbox_reg[:, -1])
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1]) * torch.sin(batched_bbox_reg[:, -1])
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                      bbox_pred=bbox_pred,
                                      bbox_dir_cls_pred=bbox_dir_cls_pred,
                                      batched_labels=batched_bbox_labels,
                                      num_cls_pos=num_cls_pos,
                                      batched_bbox_reg=batched_bbox_reg,
                                      batched_dir_labels=batched_dir_labels)

                global_step = epoch * len(val_dataloader) + val_step + 1
                if global_step % args.log_freq == 0:
                    save_summary(writer, loss_dict, global_step, 'val')
                val_step += 1
        model.train()


if __name__ == '__main__':
    main()
    # pass
