#!/usr/bin/env python
# coding: utf-8

# ### In this notebook we use [PointNet](https://arxiv.org/abs/1612.00593) to perform 3D Object Classification on
# [ModelNet40 Dataset](http://modelnet.cs.princeton.edu/#).

# <h3><center>Applications of PointNet</center></h3>
# <img src="http://stanford.edu/~rqi/pointnet/images/teaser.jpg" width="600" height="500"/>
# <h4></h4>
# <h4><center><a href="https://arxiv.org/pdf/1612.00593.pdf">Source: PointNet [Charles R. Qi et. al.]</a></center></h4>

# ## Acknowledgements
# 
# ### This work was inspired by and borrows code from [Nikita Karaev's](https://github.com/nikitakaraevv)
# [PointNet implementation](https://github.com/nikitakaraevv/pointnet).

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
from path import Path
from models.pointNet import PointNet
from models.pointNetLoss import pointnetloss
import scipy.spatial.distance
import plotly.graph_objects as go
# import plotly.express as px
import matplotlib.pyplot as plt
from CustomData.dataset import PointCloudData
from sklearn.metrics import confusion_matrix
from CustomData.dataset import read_off
from utils.draw_plot import visualize_rotate, pcshow, plot_confusion_matrix
from utils.PointProcessor import PointSampler, Normalize, ToTensor, RandomNoise, RandRotation_z
random.seed = 42
path = Path("/home/cuidongdong/data/modelnet40off/ModelNet40")

folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
classes = {folder: i for i, folder in enumerate(folders)}


with open(path / "airplane/train/airplane_0001.off", 'r') as f:
    verts, faces = read_off(f)

i, j, k = np.array(faces).T
x, y, z = np.array(verts).T
len(x)

visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='yellowgreen', opacity=0.50, i=i, j=j, k=k)]).show()

visualize_rotate([go.Scatter3d(x=x, y=y, z=z, mode='markers')]).show()
pcshow(x, y, z)
pointcloud = PointSampler(3000)((verts, faces))
pcshow(*pointcloud.T)

norm_pointcloud = Normalize()(pointcloud)
pcshow(*norm_pointcloud.T)

rot_pointcloud = RandRotation_z()(norm_pointcloud)
noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)
pcshow(*noisy_rot_pointcloud.T)
# ### Define Datasets / Dataloaders
train_transforms = transforms.Compose([
    PointSampler(1024),
    Normalize(),
    RandRotation_z(),
    RandomNoise(),
    ToTensor()
])

train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)

inv_classes = {i: cat for cat, i in train_ds.classes.items()}

print('Train dataset size: ', len(train_ds))
print('Valid dataset size: ', len(valid_ds))
print('Number of classes: ', len(train_ds.classes))
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
print('Class: ', inv_classes[train_ds[0]['category']])

train_loader = DataLoader(dataset=train_ds, batch_size=64, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

# <h3><center>PointNet Model Architecture</center></h3>
# <img src="http://stanford.edu/~rqi/pointnet/images/pointnet.jpg" width="750" height="750"/>
# <h4></h4>
# <h4><center><a href="https://arxiv.org/pdf/1612.00593.pdf">Source: PointNet [Charles R. Qi et. al.]</a></center></h4>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pointnet = PointNet()
if torch.cuda.device_count() > 1:
    print("let's use", torch.cuda.device_count(), "GPUs!")
    pointnet = nn.DataParallel(pointnet)

if torch.cuda.is_available():
    pointnet.to(device)

# Load a pre-trained model if it exists
if os.path.exists('../save.pth'):
    pointnet.load_state_dict(torch.load('../save.pth'))
    print('Loaded Pre-trained PointNet Model!')

optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.0008)


# ### Train PointNet
def train(model, train_loader, val_loader=None, epochs=1):
    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss += loss.item()
            if i % 5 == 4:  # print every 5 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / 5))
                running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # save the model
        torch.save(pointnet.state_dict(), "save.pth")


train(pointnet, train_loader, valid_loader, 30)

# ### Test PointNet
pointnet.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for i, data in enumerate(valid_loader):
        print('Batch [%4d / %4d]' % (i + 1, len(valid_loader)))

        inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
        outputs, __, __ = pointnet(inputs.transpose(1, 2))
        _, preds = torch.max(outputs.data, 1)
        all_preds += list(preds.cpu().numpy())
        all_labels += list(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(16, 16))
plot_confusion_matrix(cm, list(classes.keys()), normalize=True)
plt.figure(figsize=(16, 16))
plot_confusion_matrix(cm, list(classes.keys()), normalize=False)
