import os
import sys
import numpy as np
import scipy.io as scio
sys.path.append('../../motion')
import tools.tools as tools
import BVH as BVH
from motion.Animation import positions_global
from Animation import Animation
from Quaternions import Quaternions
from Pivots import Pivots
from InverseKinematics import BasicJacobianIK, JacobianInverseKinematics

dataset = 'mhad'
if dataset == 'hdm05':
    val_data_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_data.npy'
    core = np.load('./dataprecess/core/hdm05_core_ik.npy')
if dataset == 'mhad':
    val_data_path = '/usr/not-backed-up/dyf/code/data/mhad/val_data.npy'
    core = np.load('./dataprecess/core/mhad_core_ik.npy')
val_data = np.load(val_data_path)
targets = val_data[20, :, :, :, 0]
targets = tools.postprocessing(targets, core)
targets = np.insert(targets, 0, targets[:, 0, :], axis=1)
rest,names,_ = BVH.load('./rest.bvh')
rest2,names2,_ = BVH.load('./rest.bvh')
rest_copy = rest.copy()
index = [16,17,21,24,28]
#反向循环
for i in reversed(index):
    del names[i]
offsets = rest_copy.offsets.copy()
offsets = np.delete(offsets,[16,17,21,24,28],axis=0)
positions = offsets.reshape((1,26,3))
parents = np.array([-1,0,1,2,3,4,0,6,7,8,9,0,11,12,13,14,14,16,17,18,18,14,21,22,23,23])
#imaginaries = rest_copy.orients.imaginaries.copy()
#imaginaries = np.delete(imaginaries,[0,1,2,3,4,5],0)
orients = Quaternions.id(0)
qs = rest_copy.orients.qs.copy()
qs = np.delete(qs,[16,17,21,24,28],0)
orients.qs = qs
rotations = Quaternions.id(0)
rotations_qs = qs.reshape((1,26,4))
rotations.qs = rotations_qs
hdm05_rest = Animation(rotations, positions, orients, offsets, parents)
animik = hdm05_rest.copy()
animik.positions = animik.positions.repeat(len(targets), axis=0)
animik.rotations = animik.rotations.repeat(len(targets), axis=0)
animik.positions[:, 0, :] = targets[:, 0, :]
targetmap = {}
for ti in range(targets.shape[1]):
    targetmap[ti] = targets[:, ti]
ik = JacobianInverseKinematics(animik, targetmap, iterations=10, damping=2.0, silent=True, translate=True)
ik()
target2 = positions_global(animik)
tar_length_ori = tools.bone_length_np(np.transpose(targets[:, 1:, :], [2, 0, 1]))
tar_length_post = tools.bone_length_np(np.transpose(target2[:, 1:, :], [2, 0, 1]))
BVH.save('./hdm05_rest.bvh', hdm05_rest, names)
print('1')