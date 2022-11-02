import numpy as np

# 25 keypoints provided by the sample npy, 1-index
keypoints_to_index_25 = {'lefthip': 13, 'leftknee': 14, 'leftfoot': 15, 'lefttoe':16,
                         'righthip': 17, 'rightknee': 18, 'rightfoot': 19, 'righttoe':20,
                         'leftshoulder': 5, 'leftelbow': 6, 'leftwrist': 7, 'leftwrist2':8, 'leftthumb':23, 'lefthand':22,
                         'rightshoulder': 9, 'rightelbow': 10, 'rightwrist': 11, 'rightwrist2':12, 'rightthumb':25, 'righthand':24,
                         'hips':1, 'waist':2,
                         'neck':21, 'neckup':3, 'head': 4}

# define skeleton offset directions
hierarchy_25 = {'hips': [],
             'lefthip': ['hips'], 'leftknee': ['lefthip', 'hips'], 'leftfoot': ['leftknee', 'lefthip', 'hips'],'lefttoe': ['leftfoot', 'leftknee', 'lefthip', 'hips'],
             'righthip': ['hips'], 'rightknee': ['righthip', 'hips'], 'rightfoot': ['rightknee', 'righthip', 'hips'],'righttoe': ['rightfoot', 'rightknee', 'righthip', 'hips'],
             'neck': ['waist', 'hips'], 'neckup':['neck', 'waist', 'hips'], 'head':['neckup', 'neck', 'waist', 'hips'],
             'waist': ['hips'],
             'leftshoulder': ['neck', 'waist', 'hips'], 'leftelbow': ['leftshoulder', 'neck', 'waist','hips'], 'leftwrist': ['leftelbow', 'leftshoulder', 'neck', 'waist', 'hips'], 'leftwrist2': ['leftwrist', 'leftelbow', 'leftshoulder', 'neck', 'waist', 'hips'],
             'leftthumb': ['leftwrist2', 'leftwrist', 'leftelbow', 'leftshoulder', 'neck', 'waist', 'hips'],'lefthand': ['leftthumb', 'leftwrist2', 'leftwrist', 'leftelbow', 'leftshoulder', 'neck', 'waist', 'hips'],
             'rightthumb': ['rightwrist2', 'rightwrist', 'rightelbow', 'rightshoulder', 'neck', 'waist', 'hips'],'righthand': ['rightthumb', 'rightwrist2', 'rightwrist', 'rightelbow', 'rightshoulder', 'neck', 'waist', 'hips'],
             'rightshoulder': ['neck', 'waist', 'hips'], 'rightelbow': ['rightshoulder', 'neck', 'waist', 'hips'], 'rightwrist': ['rightelbow', 'rightshoulder', 'neck', 'waist', 'hips'],  'rightwrist2': ['rightwrist', 'rightelbow', 'rightshoulder', 'neck', 'waist', 'hips']
            }

offset_directions_25 = {'lefthip': np.array([1,0,0]),       # left lower
                        'leftknee': np.array([0,-1, 0]),
                        'leftfoot': np.array([0,-1, 0]),
                        'lefttoe': np.array([0,-1, 0]),
                        'righthip' : np.array([-1,0,0]),    # right lower
                        'rightknee' : np.array([0,-1, 0]),
                        'rightfoot' : np.array([0,-1, 0]),
                        'righttoe' : np.array([0,-1, 0]),
                        'neck' : np.array([0,1,0]),         # centrum
                        'neckup' : np.array([0,1,0]),
                        'head' : np.array([0,1,0]),
                        'waist' : np.array([0,1,0]),
                        'leftshoulder': np.array([1,0,0]),  # left upper
                        'leftelbow': np.array([1,0,0]),
                        'leftwrist': np.array([1,0,0]),
                        'lefthand': np.array([1,0,0]),
                        'leftthumb': np.array([1,0,0]),
                        'leftwrist2': np.array([1,0,0]),
                        'rightshoulder': np.array([-1,0,0]),# right upper
                        'rightelbow': np.array([-1,0,0]),
                        'rightwrist': np.array([-1,0,0]),
                        'righthand': np.array([-1,0,0]),
                        'rightthumb': np.array([-1,0,0]),
                        'rightwrist2': np.array([-1,0,0])
}

# coco is modified and would need pre-processing for data
keypoints_to_index_coco = {'lefthip': 11, 'leftknee': 12, 'leftankle': 13, 'lefteye':15, 'leftear':17,
                           'righthip': 8, 'rightknee': 9, 'rightankle': 10, 'righteye':14, 'rightear':16,
                           'leftshoulder': 5, 'leftelbow': 6, 'leftwrist': 7,
                           'rightshoulder': 2, 'rightelbow': 3, 'rightwrist': 4,
                           'hips': 18, # calculated from midpoints
                           'neck': 1, 'nose': 0}

# keypoints by PennAction dataset, 1-index
keypoints_to_index_pennaction = {'lefthip': 8, 'leftknee': 10, 'leftankle': 12,
                                 'righthip': 9, 'rightknee': 11, 'rightankle': 13,
                                 'leftshoulder': 2, 'leftelbow': 4, 'leftwrist': 6,
                                 'rightshoulder':3, 'rightelbow':5, 'rightwrist':7,
                                 'hips': 14, 'neck':15, # calculated from midpoints
                                 'head': 1}

hierarchy_pennaction = {'hips': [],
                        'lefthip': ['hips'], 'leftknee': ['lefthip', 'hips'], 'leftankle': ['leftknee', 'lefthip', 'hips'],
                        'righthip': ['hips'], 'rightknee': ['righthip', 'hips'], 'rightankle': ['rightknee', 'righthip', 'hips'],
                        'leftshoulder': ['neck', 'hips'], 'leftelbow': ['leftshoulder', 'neck','hips'], 'leftwrist': ['leftelbow', 'leftshoulder', 'neck', 'hips'],
                        'rightshoulder': ['neck', 'hips'], 'rightelbow': ['rightshoulder', 'neck', 'hips'], 'rightwrist': ['rightelbow', 'rightshoulder', 'neck', 'hips'],
                        'neck': ['hips'], 'head': ['neck', 'hips']
}

offset_directions_pennaction = {'lefthip': np.array([1,0,0]),       # left lower
                                'leftknee': np.array([0,-1, 0]),
                                'leftankle': np.array([0,-1, 0]),
                                'righthip' : np.array([-1,0,0]),    # right lower
                                'rightknee' : np.array([0,-1, 0]),
                                'rightankle' : np.array([0,-1, 0]),
                                'neck' : np.array([0,1,0]),         # centrum
                                'head' : np.array([0,1,0]),
                                'leftshoulder': np.array([1,0,0]),  # left upper
                                'leftelbow': np.array([1,0,0]),
                                'leftwrist': np.array([1,0,0]),
                                'rightshoulder': np.array([-1,0,0]),# right upper
                                'rightelbow': np.array([-1,0,0]),
                                'rightwrist': np.array([-1,0,0]),
}
   


