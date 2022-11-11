from matplotlib import pyplot as plt
import numpy as np

from data.kinematics_definition import keypoints_to_index_25 as keypoints_to_index
from data.kinematics_definition import hierarchy_25 as hierarchy

filename = 'datasets/NTU/nturgb+d_npy/S001C001P001R001A002.npy'
data = np.load(filename, allow_pickle=True).item()

n_body = data['nbodys'][0]

# import pdb; pdb.set_trace()

for i in range(n_body):
    # extract skeleton of single perosn
    data_single = data['skel_body{}'.format(i)]
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # plot skeleton using the first frame
    first_frame = data_single[0]
    for key, fathers in hierarchy.items():
        if key == 'hips': continue
        father = fathers[0]
        ax.plot3D([first_frame[keypoints_to_index[father]-1][0], first_frame[keypoints_to_index[key]-1][0]],
                  [first_frame[keypoints_to_index[father]-1][1], first_frame[keypoints_to_index[key]-1][1]],
                  [first_frame[keypoints_to_index[father]-1][2], first_frame[keypoints_to_index[key]-1][2]], 'gray')

    # plot key joint movements
    # joint_list = ['hips']
    # joint_list_idx = [v-1 for k, v in keypoints_to_index.items() if k in joint_list]
    joint_list_idx = [v-1 for k, v in keypoints_to_index.items()]

    for i in joint_list_idx:
        ax.plot3D(data_single[:, i, 0], data_single[:, i, 1], data_single[:, i, 2], linewidth=1.5)
ax.azim = 90
ax.elev = -85
ax.set_xlim3d(-0.75, 1.25)
ax.set_xlabel('x')
ax.set_ylim3d(-1, 1)
ax.set_ylabel('y')
ax.set_zlim3d(2.5, 4.5)
ax.set_zlabel('z')
plt.show()
