import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from Model import Model


# 从MVCNN里面剥离出来专门进行输入输出的验证
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11'):
        super(SVCNN, self).__init__(name)
        self.svcnntime_start = time.time()
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')

        self.vgg11 = models.vgg11(pretrained=False)
        for param in self.vgg11.parameters():
            param.requires_grad_(False)
        self.vgg11.load_state_dict(torch.load("/home/cuidongdong/pretrained_models/CV/vgg11-bbd30ac9.pth"))

        self.net_1 = self.vgg11.features
        self.net_2 = self.vgg11.classifier
        print(self.net_2)

        self.net_2._modules['6'] = nn.Linear(4096, 40)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            print('after vgg the output shape is:', y.shape)
            svctimeend = time.time()
            print('the time cost in svcnn:', svctimeend - self.svcnntime_start)
            return self.net_2(y.view(y.shape[0], -1))


class MVCNN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12):
        super(MVCNN, self).__init__(name)

        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        # self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        # self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.use_resnet = cnn_name.startswith('resnet')

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            print('after mvcnn vgg11 net_1', self.net_1)
            self.net_2 = model.net_2

    def forward(self, x):
        # 输入是[batch_size * num_view, channel, 224, 224]
        y = self.net_1(x)
        print('mvcnn shape is:', y.shape)
        y = y.view(
            (int(x.shape[0] / self.num_views), self.num_views, y.shape[-3], y.shape[-2], y.shape[-1]))  # (8,12,512,7,7)
        return self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))


if __name__ == "__main__":
    inputdata = torch.randn(4, 3, 224, 224)
    print(inputdata.shape)
    cnet = SVCNN("MVCNN", nclasses=40, pretraining=False, cnn_name="vgg11")
    output = cnet(inputdata)
    print(output.shape)

    multiinputdata = torch.randn(4, 12, 3, 224, 224)
    N, V, C, H, W = multiinputdata.size()
    in_data = Variable(multiinputdata).view(-1, C, H, W)
    cnet_2 = MVCNN("MVCNN", cnet, nclasses=40, cnn_name="vgg11", num_views=12)
    output = cnet_2(in_data)
    print(output.shape)
