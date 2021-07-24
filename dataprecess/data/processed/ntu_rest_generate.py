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



val_data_path = '/usr/not-backed-up/dyf/code/data/ntu/val_data.npy'
val_data = np.load(val_data_path)
data = val_data[:10,:,:30,:,0]
lengths = tools.bone_length_np(data,dataset='ntu_original')
data = tools.ntu_rebuild(data)
_,ntu_offsets,ntu_scale = tools.bone_length_np(data[0],dataset='ntu_rebuild',scale=True)
ntu_offsets = np.transpose(ntu_offsets,[1,0])
length = np.sqrt(np.sum(ntu_offsets**2,axis=1))
targets = val_data[0, :, :30, :, 0]
targets = tools.ntu_rebuild(targets)
targets = np.transpose(targets,[1,2,0])
targets = np.insert(targets, 0, targets[:, 0, :], axis=1)
rest,names,_ = BVH.load('./hdm05_rest.bvh')
rest2,names2,_ = BVH.load('./hdm05_rest.bvh')
rest_copy = rest.copy()
offsets_hdm05 = rest_copy.offsets.copy()
offsets = np.zeros_like(offsets_hdm05)
offsets[2:] = ntu_offsets
positions = offsets.reshape((1,26,3))
parents = np.array([-1,0,])
#parents = np.array([-1,0,1,2,3,4,1,6,7,8,1,10,11,12,11,14,15,16,17,18,11,20,21,22,23,24])
#imaginaries = rest_copy.orients.imaginaries.copy()
#imaginaries = np.delete(imaginaries,[0,1,2,3,4,5],0)
orients = Quaternions.id(0)
qs = rest_copy.orients.qs.copy()
orients.qs = qs
rotations = Quaternions.id(0)
rotations_qs = qs.reshape((1,26,4))
rotations.qs = rotations_qs
ntu_rest = Animation(rotations, positions, orients, offsets, parents)
target2 = positions_global(ntu_rest)
tar_length_ori = tools.bone_length_np(np.transpose(targets[:, 1:, :], [2, 0, 1]),'ntu_rebuild')
tar_length_post = tools.bone_length_np(np.transpose(target2[:, 1:, :], [2, 0, 1]),'ntu_rebuild')
animik = ntu_rest.copy()
animik.positions = animik.positions.repeat(len(targets), axis=0)
animik.rotations = animik.rotations.repeat(len(targets), axis=0)
#animik.positions[:, 0, :] = targets[:, 0, :]
targetmap = {}
for ti in range(targets.shape[1]):
    targetmap[ti] = targets[:, ti]
ik = JacobianInverseKinematics(animik, targetmap, iterations=300, damping=2.0, silent=True, translate=False)
ik()
target2 = positions_global(animik)
tar_length_ori = tools.bone_length_np(np.transpose(targets[:, 1:, :], [2, 0, 1]),'ntu_rebuild')
tar_length_post = tools.bone_length_np(np.transpose(target2[:, 1:, :], [2, 0, 1]),'ntu_rebuild')
BVH.save('./ntu_rest.bvh', ntu_rest, names)
print('1')