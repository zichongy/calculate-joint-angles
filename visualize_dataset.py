from matplotlib import pyplot as plt
import numpy as np

filename = 'datasets/NTU/S001C002P002R002A005.npy'
data = np.load(filename, allow_pickle=True)

fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(data.shape[1]):
    ax.plot3D(data[:, i, 0], data[:, i, 1], data[:, i, 2], linewidth=1.5)
plt.show()