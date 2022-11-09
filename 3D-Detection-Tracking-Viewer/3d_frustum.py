import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.random.uniform(10, 40, 30)
y = np.random.uniform(100, 200, 30)
z = np.random.uniform(10, 20, 30)

fig = plt.figure()
ax3d = Axes3D(fig)
ax3d.scatter(x, y, z, c="b", marker="*")

plt.show()
