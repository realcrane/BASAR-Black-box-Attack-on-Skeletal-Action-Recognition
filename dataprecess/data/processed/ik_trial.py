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
val_data_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_data.npy'
#val_data_path = '/usr/not-backed-up/dyf/code/data/ntu/xsub/val_data.npy'
val_data = np.load(val_data_path)
val_data1 = val_data[0,:,:,:,0]

rest,names,_ = BVH.load('./hdm05_rest.bvh')
core = np.load('./dataprecess/core/hdm05_core_ik.npy')
targets = tools.postprocessing(val_data1,core)
initial = np.zeros((1,3))
targets = np.insert(targets, 0, values=initial, axis=1)
animik = rest.copy()
animik.positions = animik.positions.repeat(len(targets), axis=0)
animik.rotations = animik.rotations.repeat(len(targets), axis=0)
targetmap = {}
for ti in range(targets.shape[1]):
    targetmap[ti] = targets[:, ti]
ik = JacobianInverseKinematics(animik, targetmap, iterations=50, damping=2.0, silent=True,translate=False)
ik()
target_ik = Animation.positions_global(animik)
qs = animik.rotations.qs
eulers = []
for i in range(len(qs)):
    r = R.from_quat(qs[i])
    euler =r.as_euler('xyz', degrees= True)
    eulers.append(deepcopy(euler))
eulers = np.array(eulers)
r2 = R.from_euler('xyz',euler,degrees= True)
qs_euler = r2.as_quat().reshape(1,26,4)
rest.rotations.qs = qs_euler
target_ik30 = Animation.positions_global(rest)

print('1')