

# Point_Cloud_Algorithm & 🚗 3D Object Detection & Perception

[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Task](https://img.shields.io/badge/Task-3D%20Detection-brightgreen.svg)]()

> **Seeing the world in 3D.**
> 本仓库 (`Point_Cloud_Algorithm`) 汇集了自动驾驶领域主流的 **3D 目标检测** 与 **点云深度学习** 模型的复现与学习记录。

---

## 📖 目录 (Table of Contents)

- [简介 (Introduction)](#-简介-introduction)
- [模型仓库 (Model Zoo)](#-模型仓库-model-zoo)
- [项目结构 (Project Structure)](#-项目结构-project-structure)
- [环境依赖 (Requirements)](#-环境依赖-requirements)
- [参考资料 (References)](#-参考资料-references)
- [作者 (Author)](#-作者-author)

---

## 📖 简介 (Introduction)

本项目旨在深入理解与实现自动驾驶感知栈（Perception Stack）中的核心模型。
涵盖了从早期的点云处理网络（PointNet）到当前主流的体素化检测器（PointPillars, CenterPoint），以及前沿的 BEV 视角转换算法（LSS, CaDDN）。

主要关注点：**Lidar-based Detection**, **Camera-Lidar Fusion**, **BEV Perception**.

---

## 🏛️ 模型仓库 (Model Zoo)

本项目包含以下模型的实现或源码分析，按技术流派分类：

### 1. Point-based Methods (基于点的方法)
直接处理原始点云，保留几何特征。
- **[PointNet](./PointNet)**: 开山之作，直接消耗无序点云。
- **[PointNet++](./PointNet++)**: 引入多尺度分组，提取局部特征。
- **[PointConv](./PointConv)**: 高效的点云卷积操作。

### 2. Voxel / Grid-based Methods (基于体素/网格)
将点云栅格化，利用 2D/3D CNN 高效推理，工业界主流方案。
- **[PointPillars](./PointPillars)**: 将点云柱状化 (Pillars)，速度极快，工程落地首选。
- **[CenterPoint](./CenterPoint)**: Anchor-free 范式，通过预测中心点检测 3D 物体。
- **[PIXOR](./PIXOR)**: 无高度维度的 BEV 检测器。
- **[VoxelNet/SECOND]**: (集成在相关模块中)

### 3. Camera / Multi-View / BEV (视觉与融合)
涉及图像到 3D 的转换与特征融合。
- **[LSS (Lift-Splat-Shoot)](./LSS)**: 经典的视觉 BEV 特征转换算法。
- **[CaDDN](./CaDDN)**: 深度分布网络，用于单目 3D 检测。
- **[OFT (Orthographic Feature Transform)](./OFT)**: 正交特征变换。
- **[MVCNN](./MVCNN)**: 多视图 CNN。

### 4. 2D / Anchor-based Baselines
- **[RetinaNet](./Retinanet)**: 经典的单阶段检测器。
- **[CenterNet](./CenterNet)**: Anchor-free 2D 检测。

---

## 📂 项目结构 (Project Structure)

```text
Point_Cloud_Algorithm/
├── 📂 3D-Detection-Tracking-Viewer/  # 3D 可视化工具 (基于 Open3D 或 Mayavi)
├── 📂 common_tools/                  # 通用工具库 (IoU计算, 数据增强, 坐标转换)
│
├── 🔹 Point-based
│   ├── 📂 PointNet/
│   ├── 📂 PointNet++/
│   └── 📂 PointConv/
│
├── 🔹 Voxel-based
│   ├── 📂 PointPillars/
│   ├── 📂 CenterPoint/
│   └── 📂 PIXOR/
│
├── 🔹 Camera & BEV
│   ├── 📂 LSS/              # Lift-Splat-Shoot
│   ├── 📂 CaDDN/
│   ├── 📂 OFT/
│   └── 📂 MVCNN/
│
└── 🔹 2D Baselines
    ├── 📂 CenterNet/
    └── 📂 Retinanet/

```

---

## 📦 环境依赖 (Requirements)

推荐使用 Docker 或 Anaconda 配置环境：

* **Python**: 3.7+
* **PyTorch**: 1.8+
* **CUDA**: 11.1+
* **spconv**: 2.x (For Voxel-based models)
* **Open3D / Mayavi**: For Visualization

```bash
# 安装基础依赖
pip install torch torchvision
pip install spconv-cu111  # 根据 CUDA 版本选择
pip install open3d

```

---

## 📚 参考资料 (References)

* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) - SOTA 3D toolbox
* [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
* *VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection*
* *PointPillars: Fast Encoders for Object Detection from Point Clouds*
