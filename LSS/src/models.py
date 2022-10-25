import os
import cv2
import torch
import numpy as np
import torchvision
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

# from .tools import gen_dx_bx, cumsum_trick, QuickCumsum
# 测试环境用
from tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)  # 这里使用efficientnet进行特征提取，对efficientnet输出的网络进行cat之后再卷积，输出[5, 512, 8, 22]
        # Depth
        x = self.depthnet(x)    # 就是一个卷积网络 [5, 105, 8, 22]

        depth = self.get_depth_dist(x[:, :self.D])  # 深度信息的softmax
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        # [5, 64, 41, 8, 22]
        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem input=[5,3,128,352]-->[5,32,64,176]
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        # # dx=[0.5, 0.5, 20], bx=[-49.75, -49.75, 0], nx=[200, 200, 1]
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16        # 下采样倍数
        self.camC = 64      # 图像特征通道数
        self.frustum = self.create_frustum()  # 这里就是建立一个视锥，大小是[41,8,22,3]size=[D, fH, fW, 3]
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape  # [41x1x1]->[41x8x22]
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)  # [5, 64, 41, 8, 22]
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)  # [1, 5, 64, 41, 8, 22]
        x = x.permute(0, 1, 3, 4, 5, 2)  # [1, 5, 41, 8, 22, 64]

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)  # [36080, 64]

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box         # 过滤掉在边界线之外的点 x:0~199  y: 0~199  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):  # 这里就是splat
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)   # 这里主要是利用efficientnet进行特征提取

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)


def one_single_data():

    normalize_img = torchvision.transforms.Compose((
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ))

    dataroot = '/home/cuidongdong/nuscenes_mini'
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=dataroot,
                    verbose=False)
    data = nusc.get("sample", "965f6af5a92449348409029a5f048a38")
    H = 900,  # 图片大小
    W = 1600,
    resize_lim = (0.193, 0.225),  # resize的范围
    final_dim = (128, 352),  # 数据预处理之后图片最后的尺寸
    bot_pct_lim = (0.0, 0.22),  # 裁剪图片时，图像底部裁减掉部分所占的比例
    rot_lim = (-5.4, 5.4),  # 训练时旋转图片的角度范围
    rand_flip = True,  # 随机翻转
    ncams = 5,  # 摄像机通道数
    max_grad_norm = 5.0,
    pos_weight = 2.13,  # 损失函数中给正样本项损失乘的权重系数
    logdir = './runs',  # 日志的输出文件

    xbound = [-50.0, 50.0, 0.5],  # 限制x的方向范围并进行划分网格
    ybound = [-50.0, 50.0, 0.5],  # 限制y的方向范围并进行划分网格
    zbound = [-10.0, 10.0, 20.0],  # 限制z的方向范围并进行划分网格
    dbound = [4.0, 45.0, 1.0],  # 限制深度方向范围并进行划分网格

    grid_conf = {  # 网格配置
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],  # 限制z的方向范围并进行划分网格,
        'dbound': [4.0, 45.0, 1.0],  # 限制深度方向范围并进行划分网格,
    }
    data_aug_conf = {  # 数据增强配置
        'resize_lim': (0.193, 0.225),  # resize的范围,
        'final_dim': (128, 352),
        'rot_lim': (-5.4, 5.4),  # 训练时旋转图片的角度范围,
        'H': 900, 'W': 1600,
        'rand_flip': True,
        'bot_pct_lim': (0.0, 0.22),  # 裁剪图片时，图像底部裁减掉部分所占的比例,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_BACK']
    is_train =True

    def sample_augmentation():
        H, W = data_aug_conf['H'], data_aug_conf['W']
        fH, fW = data_aug_conf['final_dim']
        print(fH, fW)
        if is_train:
            resize = np.random.uniform(*data_aug_conf['resize_lim'])  # 随机取resize的一个范围
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_rot(h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        #
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran


    rec = data
    imgs = []
    rots = []
    trans = []
    intrins = []
    post_rots = []
    post_trans = []
    is_train = True
    for cam in cams:
        samp = nusc.get('sample_data', rec['data'][cam])  # 拿到该摄像头的token对应的数据
        imgname = os.path.join(nusc.dataroot, samp['filename'])
        img = Image.open(imgname)
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        # 获取该摄像头的标定数据
        sens = nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
        intrin = torch.Tensor(sens['camera_intrinsic'])  # 拿到相机的内参
        rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)  # 传感器的旋转参角
        tran = torch.Tensor(sens['translation'])  # 相加外参，偏移矩阵

        # augmentation (resize, crop, horizontal flip, rotate)
        resize, resize_dims, crop, flip, rotate = sample_augmentation()  # 数据增强
        img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                   resize=resize,
                                                   resize_dims=resize_dims,
                                                   crop=crop,
                                                   flip=flip,
                                                   rotate=rotate,
                                                   )

        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2

        imgs.append(normalize_img(img))
        intrins.append(intrin)
        rots.append(rot)
        trans.append(tran)
        post_rots.append(post_rot)
        post_trans.append(post_tran)

    return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
            torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))  # 使用torch.stack组装到一起


if __name__ == "__main__":

    nepochs = 10000,
    gpuid = 1,
    H = 900,  # 图片大小
    W = 1600,
    resize_lim = (0.193, 0.225),  # resize的范围
    final_dim = (128, 352),  # 数据预处理之后图片最后的尺寸
    bot_pct_lim = (0.0, 0.22),  # 裁剪图片时，图像底部裁减掉部分所占的比例
    rot_lim = (-5.4, 5.4),  # 训练时旋转图片的角度范围
    rand_flip = True,  # 随机翻转
    ncams = 5,  # 摄像机通道数
    max_grad_norm = 5.0,
    pos_weight = 2.13,  # 损失函数中给正样本项损失乘的权重系数
    logdir = './runs',  # 日志的输出文件

    xbound = [-50.0, 50.0, 0.5],  # 限制x的方向范围并进行划分网格
    ybound = [-50.0, 50.0, 0.5],  # 限制y的方向范围并进行划分网格
    zbound = [-10.0, 10.0, 20.0],  # 限制z的方向范围并进行划分网格
    dbound = [4.0, 45.0, 1.0],  # 限制深度方向范围并进行划分网格

    grid_conf = {  # 网格配置
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],  # 限制z的方向范围并进行划分网格,
        'dbound': [4.0, 45.0, 1.0],  # 限制深度方向范围并进行划分网格,
    }
    data_aug_conf = {  # 数据增强配置
        'resize_lim': (0.193, 0.225),  # resize的范围,
        'final_dim': (128, 352),
        'rot_lim': (-5.4, 5.4),  # 训练时旋转图片的角度范围,
        'H': 900, 'W': 1600,
        'rand_flip': True,
        'bot_pct_lim': (0.0, 0.22),  # 裁剪图片时，图像底部裁减掉部分所占的比例,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }

    imgs, rots, trans, intrins, post_rots, post_trans = one_single_data()
    imgs = imgs.unsqueeze(0)
    rots = rots.unsqueeze(0)
    trans = trans.unsqueeze(0)
    intrins = intrins.unsqueeze(0)
    post_rots = post_rots.unsqueeze(0)
    post_trans = post_trans.unsqueeze(0)
    # print(imgs.shape)
    # device = torch.device('cpu') if gpuid < 0 else torch.device('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = compile_model(grid_conf, data_aug_conf, outC=1)

    model.to(device)

    preds = model(imgs.to(device),
                  rots.to(device),
                  trans.to(device),
                  intrins.to(device),
                  post_rots.to(device),
                  post_trans.to(device),
                  )
    # binimgs = binimgs.to(device)
    print(preds.shape)