import os
import sys
import numpy as np
import scipy.io as scio
sys.path.append('../../motion')
from Animation import Animation
import pickle
import motion.Animation as Animation
from Quaternions import Quaternions
import tools.kinematics as kinematics
import motion.BVH as BVH

import tools.tools as tools
from motion.InverseKinematics import JacobianInverseKinematics

val_label_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_label.pkl'
with open(val_label_path, 'rb') as f:
    sample = pickle.load(f)
val_data_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_data.npy'
#val_data_path = '/usr/not-backed-up/dyf/code/data/ntu/xsub/val_data.npy'
val_data = np.load(val_data_path)
val_data1 = val_data[0,:,:,:,0]
rest,names,_ = BVH.load('./rest.bvh')
rest2,names2,_ = BVH.load('./hdm05_rest.bvh')
initial_target = Animation.positions_global(rest)
hdm05anim,_,_ = BVH.load('./HDM_bd_cartwheelLHandStart1Reps_001_120.bvh')
positions = Animation.positions_global(hdm05anim)
targets = positions
anim = rest.copy()
anim.positions = anim.positions.repeat(len(targets), axis=0)
anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
anim.positions[:, 0] = targets[:, 0]
anim.rotations[:, :] = hdm05anim.rotations[:, :]

targetmap = {}
for ti in range(targets.shape[1]):
    targetmap[ti] = targets[:, ti]

ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=True)
ik()
positions_ik = Animation.positions_global(anim)
#ik
targets = np.delete(positions,[0,16,17,21,24,28],axis=1)
target_sample_ik,target_sample_euler = kinematics.target_ik(targets,translate=False)
positions_ik = Animation.positions_global(target_sample_ik)
positions2 = np.transpose(targets,[0,2,1])
matlab_data = positions2.reshape((1,401,75), order='F')



import scipy.io as io
io.savemat('./1.mat', {'data': matlab_data})
print('1')