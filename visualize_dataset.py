from matplotlib import pyplot as plt
import numpy as np

from data.kinematics_definition import keypoints_to_index_25 as keypoints_to_index
from data.kinematics_definition import hierarchy_25 as hierarchy

filename = 'datasets/NTU/nturgb+d_npy/S001C001P001R001A001.npy'
data = np.load(filename, allow_pickle=True).item()

n_body = data['nbodys'][0]

# import pdb; pdb.set_trace()

for i in range(n_body):
    # extract skeleton of single perosn
    data_single = data['skel_body{}'.format(i)]
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # plot skeleton using the first frame
    first_person = data_single[0]
    for key, fathers in hierarchy.items():
        if key == 'hips': continue
        father = fathers[0]
        ax.plot3D([first_person[keypoints_to_index[father]-1][0], first_person[keypoints_to_index[key]-1][0]],
                  [first_person[keypoints_to_index[father]-1][1], first_person[keypoints_to_index[key]-1][1]],
                  [first_person[keypoints_to_index[father]-1][2], first_person[keypoints_to_index[key]-1][2]], 'gray')

    # plot joint movements
    for i in range(data_single.shape[1]):
        ax.plot3D(data_single[:, i, 0], data_single[:, i, 1], data_single[:, i, 2], linewidth=1.5)
plt.show()
