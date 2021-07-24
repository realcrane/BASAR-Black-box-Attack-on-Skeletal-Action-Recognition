import os
import sys
import numpy as np
import scipy.io as scio
from copy import deepcopy
sys.path.append('../../motion')
from motion.Quaternions import Quaternions
import motion.BVH as BVH
import motion.Animation as Animation
import tools.tools as tools
from tools.bone_constraints import bone_constraints_np
from motion.InverseKinematics import JacobianInverseKinematics
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('../motion')
from Quaternions import Quaternions
import BVH as BVH
import Animation as Animation
import tools.tools as tools
from tools.bone_constraints import bone_constraints_np
import tools.kinematics as kinematics
val_data_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_data.npy'
#val_data_path = '/usr/not-backed-up/dyf/code/data/ntu/xsub/val_data.npy'
val_data = np.load(val_data_path)
val_data1 = val_data[0,:,:,:,0]
rest,names,_ = BVH.load('./hdm05_rest.bvh')
rest_or,names_or,_ = BVH.load('./rest.bvh')
#rest,names,_ = BVH.load('./hdm05_rest.bvh')
core = np.load('./dataprecess/core/hdm05_core_ik.npy')
target_process = tools.postprocessing(val_data1,core)
target_sample_ik,target_sample_euler = kinematics.target_ik(target_process)
target_sample_ik_process = Animation.positions_global(target_sample_ik)
target_process = np.transpose(target_process, [2,0,1])
target_sample_ik_process = np.transpose(target_sample_ik_process[:,1:,:],[2,0,1])
link,target_ik_length = tools.bone_length_np(target_sample_ik_process)
_,target_length = tools.bone_length_np(target_process)
rest_or_process = Animation.positions_global(rest_or)
rest_process = Animation.positions_global(rest)
target_length_ik = Animation.position_lengths(target_sample_ik)

print('1')