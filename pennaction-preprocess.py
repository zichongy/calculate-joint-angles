import numpy as np
import scipy.io as sio
from os import listdir


N_JOINTS = 13
N_JOINTS_NEW = 15
label_dir = 'datasets/Penn_Action/labels/'
output_dir = 'datasets/Penn_Action/labels_npy/'


for file in listdir(label_dir):
    filename = label_dir + file
    mat = sio.loadmat(filename)

    n_frames = mat['nframes'][0][0]

    # extract pose pixel location and save as npy file
    pose_xyz = np.zeros((3, n_frames, N_JOINTS_NEW))
    pose_xyz[0,:,:N_JOINTS] = mat['x']
    pose_xyz[1,:,:N_JOINTS] = mat['y']
    pose_xyz[:,:,13] = (pose_xyz[:,:,7] + pose_xyz[:,:,8]) / 2
    pose_xyz[:,:,14] = (pose_xyz[:,:,1] + pose_xyz[:,:,2]) / 2
    # bbox = mat['bbox']
    out_filename = output_dir + file.replace('.mat', '.npy')
    np.save(out_filename, pose_xyz)