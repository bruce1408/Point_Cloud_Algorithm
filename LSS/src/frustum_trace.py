# -*- coding:utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def trans_base(x, y, z):
    # 计算三维空间中的旋转矩阵
    Txyz = np.array(
        [
            [np.cos(y) * np.cos(z), -np.cos(y) * np.sin(z), -np.sin(y), 0],
            [np.cos(x) * np.sin(z) - np.sin(x) * np.sin(y) * np.cos(z),
             np.cos(x) * np.cos(z) + np.sin(x) * np.sin(y) * np.sin(z),
             -np.sin(x) * np.cos(y), 0
             ],
            [np.sin(x) * np.sin(z) + np.cos(x) * np.sin(y) * np.cos(z),
             np.sin(x) * np.cos(z) - np.cos(x) * np.sin(y) * np.sin(z),
             np.cos(x) * np.cos(y), 0
             ],
            [0, 0, 0, 1]
        ]
    )
    return Txyz


def gen_cube(a=1):
    # 生成一个以原点为中心,边长为a的正方体
    d = []
    X = np.linspace(a / 2, -a / 2, 2)
    for x1 in X:
        for x2 in X:
            for x3 in X:
                d.append([x1, x2, x3])
    return np.array(d)


def to_hc(a):
    # 齐次坐标表示矩阵
    a = np.column_stack((a, np.zeros(a.shape[0])))
    r = np.zeros(a.shape[1])
    r[-1] = 1
    a = np.row_stack((a, r))
    return a


def trans_cube(cube, x=0, y=0, z=0):
    # 旋转立方体
    x_ = x / 180 * np.pi
    y_ = y / 180 * np.pi
    z_ = z / 180 * np.pi
    T = trans_base(x_, y_, z_)
    cube2 = to_hc(cube)
    c2 = np.dot(T, cube2.T).T
    return c2[:-1, :-1]


def map_2d(cube, z0):
    # 在二维平面上计算视锥体投影
    cube_ = cube.copy()
    z0_ = max(z0, np.abs(cube_[:, 2].min()))  # 确保立方体不会被成像屏幕穿过
    cube_[:, 2] = cube_[:, 2] + z0_  # 平移立方体
    cube2 = np.dot(np.diag(z0_ / cube_[:, 2]), cube_)
    l = []
    for i in range(len(cube_)):
        for j in range(len(cube_)):
            d = cube_[i] - cube_[j]
            if d.dot(d) / 1 - 1 < 1E-10:
                tc = np.array([cube2[i], cube2[j]])
                l.append(tc)
    return cube2, l


def plot_2d(cube, thetax, thetay, thetaz, d, color):
    # 旋转并投影
    cube_ = trans_cube(cube, thetax, thetay, thetaz)  # 为了更加生动，对立方体做了旋转
    cube_t = np.dot(trans_cube(cube_, -90, 0, 0), np.diag([1, 1, -1]))  # 从物体空间映射至视点空间
    cube_2d, edges = map_2d(cube_t, d)
    plt.scatter(x=cube_2d[:, 0], y=cube_2d[:, 1], color=color)
    for e in edges:
        plt.plot(e[:, 0], e[:, 1], c='grey', alpha=0.3)
    plt.show()


def plot_3d(cube, thetax, thetay, thetaz, color):
    # 立方体矩阵
    cube_ = trans_cube(cube, thetax, thetay, thetaz)  # 为了更加生动，对立方体做了旋转
    x, y, z = cube_[:, 0], cube_[:, 1], cube_[:, 2]
    fig = plt.figure()
    ax2 = Axes3D(fig)
    ax2.scatter3D(x, y, z, color=color, s=30)
    for i in range(len(cube_)):
        for j in range(len(cube_)):
            # 绘制棱
            d = cube_[i] - cube_[j]
            if d.dot(d) / 1 - 1 < 1E-10:
                tc = np.array([cube_[i], cube_[j]])
                tx, ty, tz = tc[:, 0], tc[:, 1], tc[:, 2]
                ax2.plot3D(tx, ty, tz, alpha=0.3, c='grey', lw=2)
    fig.show()


# 定义正方体顶角的颜色
color = ['red', 'orange', 'limegreen', 'cyan', 'blue', 'royalblue', 'purple', 'deeppink']

# 生成立方体
cube = gen_cube(a=1)
print(cube.shape)

# 绘图
plot_3d(cube, 45, 45, 45, color)
# plot_2d(cube, 45, 45, 45, 1.5, color)  # 近距离投影
plot_2d(cube, 45, 45, 45, 100, color)  # 远距离投影