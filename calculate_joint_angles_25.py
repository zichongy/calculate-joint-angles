# from symbol import import_stmt
import numpy as np
import sys
import utils.utils as utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data.kinematics_definition import keypoints_to_index_25 as keypoints_to_index
from data.kinematics_definition import hierarchy_25 as hierarchy
from data.kinematics_definition import offset_directions_25 as offset_directions

# def read_keypoints(filename_xyz, filename_angle):

#     num_keypoints = 25
#     kpts_angle = np.load(filename_angle, allow_pickle=True)
#     kpts_angle = np.transpose(kpts_angle, axes=[1, 2, 0])

#     kpts_xyz = np.load(filename_xyz, allow_pickle=True)
#     kpts_xyz = np.transpose(kpts_xyz, axes=[1, 2, 0])
#     # kpts_xyz[:, :, 2] = np.zeros((50, 25))
#     # import pdb; pdb.set_trace()
#     return kpts_xyz, kpts_angle

def read_keypoints(filename_xyz):

    num_keypoints = 25
    kpts_xyz = np.load(filename_xyz, allow_pickle=True)
    # kpts_xyz = np.transpose(kpts_xyz, axes=[1, 2, 0])
    # kpts_xyz[:, :, 2] = np.zeros((50, 25))
    # import pdb; pdb.set_trace()
    return kpts_xyz


def convert_to_dictionary(kpts):
    kpts_dict = {}
    for key, k_index in keypoints_to_index.items():
        k_index = k_index-1
        kpts_dict[key] = kpts[:,k_index]

    kpts_dict['joints'] = list(keypoints_to_index.keys())

    return kpts_dict

def assign_joint_angles(kpts, angles):
    for joint in kpts['joints']:
        keypoint = joint+'_angles'
        # import pdb; pdb.set_trace()
        index = keypoints_to_index[keypoint.replace("_angles","")]
        kpts[joint+'_angles'] = np.array(angles[:, index-1, :])
    return kpts


def add_hips_and_neck(kpts):
    #we add two new keypoints which are the mid point between the hips and mid point between the shoulders

    # #add hips kpts
    # difference = kpts['lefthip'] - kpts['righthip']
    # difference = difference/2
    # hips = kpts['righthip'] + difference
    # kpts['hips'] = hips
    # kpts['joints'].append('hips')
    #
    #
    # #add neck kpts
    # difference = kpts['leftshoulder'] - kpts['rightshoulder']
    # difference = difference/2
    # neck = kpts['rightshoulder'] + difference
    # kpts['neck'] = neck
    # kpts['joints'].append('neck')
    kpts['hierarchy'] = hierarchy
    kpts['root_joint'] = 'hips'

    return kpts

#remove jittery keypoints by applying a median filter along each axis
def median_filter(kpts, window_size = 3):

    import copy
    filtered = copy.deepcopy(kpts)

    from scipy.signal import medfilt


    #apply median filter to get rid of poor keypoints estimations
    for joint in filtered['joints']:
        joint_kpts = filtered[joint]
        xs = joint_kpts[:,0]
        ys = joint_kpts[:,1]
        zs = joint_kpts[:,2]
        xs = medfilt(xs, window_size)
        ys = medfilt(ys, window_size)
        zs = medfilt(zs, window_size)
        filtered[joint] = np.stack([xs, ys, zs], axis = -1)

    return filtered

def get_bone_lengths(kpts):

    """
    We have to define an initial skeleton pose(T pose).
    In this case we need to known the length of each bone.
    Here we calculate the length of each bone from data
    """

    bone_lengths = {}
    for joint in kpts['joints']:
        if joint == 'hips': continue
        parent = kpts['hierarchy'][joint][0]

        joint_kpts = kpts[joint]
        parent_kpts = kpts[parent]

        _bone = joint_kpts - parent_kpts
        _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis = -1))

        _bone_length = np.median(_bone_lengths)
        bone_lengths[joint] = _bone_length

        # plt.hist(bone_lengths, bins = 25)
        # plt.title(joint)
        # plt.show()

    #print(bone_lengths)
    kpts['bone_lengths'] = bone_lengths
    return

#Here we define the T pose and we normalize the T pose by the length of the hips to neck distance.
def get_base_skeleton(kpts, normalization_bone = 'neck'):

    #this defines a generic skeleton to which we can apply rotations to
    body_lengths = kpts['bone_lengths']

    #set bone normalization length. Set to 1 if you dont want normalization
    normalization = kpts['bone_lengths'][normalization_bone]
    # normalization = 1


    #base skeleton set by multiplying offset directions by measured bone lengths. In this case we use the average of two sided limbs. E.g left and right hip averaged
    base_skeleton = {'hips': np.array([0,0,0])}
    def _set_length(joint_type):
        base_skeleton['left' + joint_type] = offset_directions['left' + joint_type] * ((body_lengths['left' + joint_type] + body_lengths['right' + joint_type])/(2 * normalization))
        base_skeleton['right' + joint_type] = offset_directions['right' + joint_type] * ((body_lengths['left' + joint_type] + body_lengths['right' + joint_type])/(2 * normalization))

    _set_length('hip')
    _set_length('knee')
    _set_length('foot')
    _set_length('toe')
    _set_length('shoulder')
    _set_length('elbow')
    _set_length('wrist')
    _set_length('wrist2')
    _set_length('hand')
    _set_length('thumb')
    base_skeleton['neck'] = offset_directions['neck'] * (body_lengths['neck']/normalization)
    base_skeleton['neckup'] = offset_directions['neckup'] * (body_lengths['neckup']/normalization)
    base_skeleton['head'] = offset_directions['head'] * (body_lengths['head']/normalization)
    base_skeleton['waist'] = offset_directions['waist'] * (body_lengths['waist']/normalization)

    kpts['offset_directions'] = offset_directions
    kpts['base_skeleton'] = base_skeleton
    kpts['normalization'] = normalization

    return



def get_hips_position_and_rotation(frame_pos, root_joint = 'hips', root_define_joints = ['lefthip', 'neck']):
    """
    Calculates the rotation of the root joint with respect to the world coordinates.
    
    :param frame_pos: The positions of the joints of a single frame
    """

    #root position is saved directly
    root_position = frame_pos[root_joint]

    #calculate unit vectors of root joint
    root_u = frame_pos[root_define_joints[0]] - frame_pos[root_joint]
    root_u = root_u/np.sqrt(np.sum(np.square(root_u)))
    root_v = frame_pos[root_define_joints[1]] - frame_pos[root_joint]
    root_v = root_v/np.sqrt(np.sum(np.square(root_v)))
    root_w = np.cross(root_u, root_v)

    #Make the rotation matrix
    C = np.array([root_u, root_v, root_w]).T
    thetaz,thetay, thetax = utils.Decompose_R_ZXY(C)
    root_rotation = np.array([thetaz, thetax, thetay])

    return root_position, root_rotation

#calculate the rotation matrix and joint angles input joint
def get_joint_rotations(joint_name, joints_hierarchy, joints_offsets, frame_rotations, frame_pos):

    _invR = np.eye(3)
    for i, parent_name in enumerate(joints_hierarchy[joint_name]):
        if i == 0: continue
        _r_angles = frame_rotations[parent_name]
        R = utils.get_R_z(_r_angles[0]) @ utils.get_R_x(_r_angles[1]) @ utils.get_R_y(_r_angles[2])
        _invR = _invR@R.T

    b = _invR @ (frame_pos[joint_name] - frame_pos[joints_hierarchy[joint_name][0]])
    # print(joint_name)
    _R = utils.Get_R2(joints_offsets[joint_name], b)
    tz, ty, tx = utils.Decompose_R_ZXY(_R)
    joint_rs = np.array([tz, tx, ty])
    #print(np.degrees(joint_rs))

    return joint_rs

#helper function that composes a chain of rotation matrices
def get_rotation_chain(joint, hierarchy, frame_rotations):

    hierarchy = hierarchy[::-1]

    #this code assumes ZXY rotation order
    R = np.eye(3)
    for parent in hierarchy:
        angles = frame_rotations[parent]
        _R = utils.get_R_z(angles[0])@utils.get_R_x(angles[1])@utils.get_R_y(angles[2])
        R = R @ _R

    return R

#calculate the joint angles frame by frame.
def calculate_joint_angles(kpts):

    #set up emtpy container for joint angles
    for joint in kpts['joints']:
        kpts[joint+'_angles'] = []

    for framenum in range(kpts['hips'].shape[0]):

        #get the keypoints positions in the current frame
        frame_pos = {}
        for joint in kpts['joints']:
            frame_pos[joint] = kpts[joint][framenum]

        root_position, root_rotation = get_hips_position_and_rotation(frame_pos)

        frame_rotations = {'hips': root_rotation}

        #center the body pose
        for joint in kpts['joints']:
            frame_pos[joint] = frame_pos[joint] - root_position

        #get the max joints connection
        max_connected_joints = 0
        for joint in kpts['joints']:
            if len(kpts['hierarchy'][joint]) > max_connected_joints:
                max_connected_joints = len(kpts['hierarchy'][joint])

        depth = 2
        while(depth <= max_connected_joints):
            for joint in kpts['joints']:
                if len(kpts['hierarchy'][joint]) == depth:
                    joint_rs = get_joint_rotations(joint, kpts['hierarchy'], kpts['offset_directions'], frame_rotations, frame_pos)
                    parent = kpts['hierarchy'][joint][0]
                    frame_rotations[parent] = joint_rs
            depth += 1

        #for completeness, add zero rotation angles for endpoints. This is not necessary as they are never used.
        for _j in kpts['joints']:
            if _j not in list(frame_rotations.keys()):
                frame_rotations[_j] = np.array([0.,0.,0.])

        #update dictionary with current angles.
        for joint in kpts['joints']:
            kpts[joint + '_angles'].append(frame_rotations[joint])


    #convert joint angles list to numpy arrays.
    for joint in kpts['joints']:
        kpts[joint+'_angles'] = np.array(kpts[joint + '_angles'])
        #print(joint, kpts[joint+'_angles'].shape)

    return

#draw the pose from original data
def draw_skeleton_from_joint_coordinates(kpts):

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')

    for framenum in range(kpts['lefthip'].shape[0]):
        #kpts['lefthip'].shape[0]
        # print(framenum)
        # if framenum%2 == 0: continue #skip every 2nd frame
        frame_rotations = {}
        for joint in kpts['joints']:
            # print(kpts[joint+'_angles'].shape)
            frame_rotations[joint] = kpts[joint+'_angles'][framenum]
        for _j in kpts['joints']:
            if _j == 'hips': continue
            _p = kpts['hierarchy'][_j][0] #get the name of the parent joint
            r1 = kpts[_p][framenum]
            r2 = kpts[_j][framenum]
            # import pdb; pdb.set_trace()
            plt.plot(xs = [r1[0], r2[0]], ys = [r1[1], r2[1]], zs = [r1[2], r2[2]], color = 'blue')
            ax.text(r2[0],r2[1],r2[2],  '%s' % (str(np.round(frame_rotations[_j]*180/np.pi,2))), size=7, zorder=1, weight='bold', color='k')

        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-0.8, 0.8)
        ax.set_xlabel('x')
        ax.set_ylim3d(-0.8, 0.8)
        ax.set_ylabel('y')
        ax.set_zlim3d(-0.8, 0.8)
        ax.set_zlabel('z')
        plt.pause(0.2)
        plt.waitforbuttonpress()
        ax.cla()
    plt.close()

#recalculate joint positions from calculated joint angles and draw
def draw_skeleton_from_joint_angles(kpts):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')

    for framenum in range(kpts['hips'].shape[0]):

        #get a dictionary containing the rotations for the current frame
        frame_rotations = {}
        for joint in kpts['joints']:
            frame_rotations[joint] = kpts[joint+'_angles'][framenum]

        #for plotting
        for _j in kpts['joints']:
            if _j == 'hips': continue

            #get hierarchy of how the joint connects back to root joint
            hierarchy = kpts['hierarchy'][_j]

            #get the current position of the parent joint
            r1 = kpts['hips'][framenum]/kpts['normalization']
            for parent in hierarchy:
                if parent == 'hips': continue
                R = get_rotation_chain(parent, kpts['hierarchy'][parent], frame_rotations)
                r1 = r1 + R @ kpts['base_skeleton'][parent]

            #get the current position of the joint. Note: r2 is the final position of the joint. r1 is simply calculated for plotting.
            r2 = r1 + get_rotation_chain(hierarchy[0], hierarchy, frame_rotations) @ kpts['base_skeleton'][_j]

            plt.plot(xs = [r1[0], r2[0]], ys = [r1[1], r2[1]], zs = [r1[2], r2[2]], color = 'red')
            ax.text(r2[0],r2[1], r2[2],  '%s' % (str(np.round(frame_rotations[_j]*180/np.pi,2))), size=7, zorder=1, weight='bold', color='k')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.azim = 90
        ax.elev = -85
        ax.set_title('Pose from joint angles')
        ax.set_xlim3d(-5, 5)
        ax.set_xlabel('x')
        ax.set_ylim3d(-5, 5)
        ax.set_ylabel('y')
        ax.set_zlim3d(-5, 5)
        ax.set_zlabel('z')
        plt.pause(0.2)
        plt.waitforbuttonpress()
        ax.cla()
    plt.close()

if __name__ == '__main__':

    # if len(sys.argv) != 3:
    #     print('Call program with input pose file')
    #     quit()

    # load the pose file
    # filename_xyz = sys.argv[1]
    filename_xyz = 'datasets/NTU/S001C002P002R002A005.npy'
    # filename_angle = sys.argv[2]
    # kpts, kpts_angle = read_keypoints(filename_xyz, filename_angle)
    kpts = read_keypoints(filename_xyz)

    #record time
    import time
    start = time.time()

    #rotate to orient the pose better
    # R = utils.get_R_z(np.pi/2)
    # for framenum in range(kpts.shape[0]):
    #     for kpt_num in range(kpts.shape[1]):
    #         kpts[framenum,kpt_num] = R @ kpts[framenum,kpt_num]

    # convert to dictionary of joints, each key stores cooordiantes of all the frames of that joint
    kpts = convert_to_dictionary(kpts)

    # define the hierarchy and root joint
    add_hips_and_neck(kpts)

    # apply median filter, per joint, per axis
    filtered_kpts = median_filter(kpts)

    # calculate bone lengths by finding median distance between joints
    get_bone_lengths(filtered_kpts)

    # symmetrize and normalize
    get_base_skeleton(filtered_kpts)

    # add original angles to the dictionary
    # filtered_kpts_assign = assign_joint_angles(filtered_kpts, kpts_angle)
    
    # calculate joint angles based on processed skeleton
    calculate_joint_angles(filtered_kpts)
    
    # record time taken
    end = time.time()
    print("time: ", end-start)

    # draw the skeleton
    # draw_skeleton_from_joint_coordinates(filtered_kpts_assign)
    draw_skeleton_from_joint_angles(filtered_kpts)
