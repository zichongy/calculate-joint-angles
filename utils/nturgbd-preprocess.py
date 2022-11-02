"""
Data loading is mainly based on https://github.com/shahroudy/NTURGB-D

Some useful information listed below is also from the above link:

Each sample is saved as the array in numpy with name like SxxxCxxxPxxxRxxxAxxx.npy
You can read the data by:
data = np.load('./SxxxCxxxPxxxRxxxAxxx.npy',allow_pickle=True).item()

file_name:      file's name
nbodys:         it's a list with same length of the sequence. it represents the number of the actors in each frame.
njoints:        the number of the joint node in the skeleton, it's a constant here
skel_bodyx:     the skeleton coordinate with the shape of (nframe, njoint, 3), the x denotes the id of the acting person in each frame.
rgb_bodyx:      the projection of the skeleton coordinate in RGBs.
depth_bodyx:    the projection of the skeleton coordinate in Depths
"""

import numpy as np
import os
import sys

load_txt_path = 'datasets/NTU/nturgb+d_skeletons/'
save_npy_path = 'datasets/NTU/nturgb+d_npy/'
missing_file_path = 'datasets/NTU/NTU_RGBD_samples_with_missing_skeletons.txt'
# missing_file_path' = 'datasets/NTU/NTU_RGBD120_samples_with_missing_skeletons.txt'
step_ranges = list(range(0,100)) # just parse range, for the purpose of paralle running.


toolbar_width = 50
def _print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def _end_toolbar():
    sys.stdout.write('\n')

def _load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True 
    return missing_files 

def _read_skeleton(file_path, save_skelxyz=True, save_rgbxy=False, save_depthxy=False):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the 
    # abundant bodys. 
    # read all lines into the pool to speed up, less io operation. 
    nframe = int(datas[0][:-1])
    bodymat = dict()
    # bodymat['file_name'] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat['nbodys'] = [] 
    bodymat['njoints'] = njoints 
    for body in range(max_body):
        if save_skelxyz:
            bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])    
        if bodycount == 0:
            continue 
        # skip the empty frame 
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)
            
            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1
            
            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame,joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame,joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame,joint] = jointinfo[5:7]
    # prune the abundant bodys 
    for each in range(max_body):
        if each >= max(bodymat['nbodys']):
            if save_skelxyz:
                del bodymat['skel_body{}'.format(each)]
            if save_rgbxy:
                del bodymat['rgb_body{}'.format(each)]
            if save_depthxy:
                del bodymat['depth_body{}'.format(each)]
    return bodymat 

        
if __name__ == '__main__':
    missing_files = _load_missing_file(missing_file_path)
    datalist = os.listdir(load_txt_path)
    alread_exist = os.listdir(save_npy_path)
    alread_exist_dict = dict(zip(alread_exist, len(alread_exist) * [True]))
     
    for ind, each in enumerate(datalist):
        _print_toolbar(ind * 1.0 / len(datalist),
                       '({:>5}/{:<5})'.format(
                           ind + 1, len(datalist)
                       ))
        S = int(each[1:4])
        if S not in step_ranges:
            continue 
        if each+'.npy' in alread_exist_dict:
            print('file already existed !')
            continue
        if each[:20] in missing_files:
            print('file missing')
            continue 
        loadname = load_txt_path+each
        print(each)
        mat = _read_skeleton(loadname)
        mat = np.array(mat)
        save_path = save_npy_path+'{}.npy'.format(each.replace('.skeleton',''))
        np.save(save_path, mat)
        # raise ValueError()
    _end_toolbar()