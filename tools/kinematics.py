import os
import sys
import numpy as np
sys.path.append('../motion/')
from motion.Quaternions import Quaternions
import motion.BVH as BVH
import motion.Animation as Animation
import tools.tools as tools
from tools.bone_constraints import bone_constraints_np
from motion.InverseKinematics import JacobianInverseKinematics
from scipy.spatial.transform import Rotation as R
from copy import deepcopy


def target_ik(targets,translate = True,num = 200,dataset=''):#dataset = 'hdm05'
    if dataset == 'hdm05' or dataset == 'mhad':
        rest, names, _ = BVH.load('../data/processed/hdm05_rest.bvh')
    elif dataset == 'ntu':
        rest,_,_ = BVH.load('../data/processed/ntu_rest.bvh')

    if len(targets[0,:,0])==25:
        targets = np.insert(targets, 0, targets[:, 0, :], axis=1)
    animik = rest.copy()
    animik.positions = animik.positions.repeat(len(targets), axis=0)
    animik.rotations = animik.rotations.repeat(len(targets), axis=0)
    animik.positions[:, 0, :] = targets[:, 0, :]
    targetmap = {}
    for ti in range(targets.shape[1]):
        targetmap[ti] = targets[:, ti]
    ik = JacobianInverseKinematics(animik, targetmap, iterations=num, damping=2.0, silent=True, translate=translate)
    ik()
    '''
    if dataset == 'ntu' and translate == True:
        positions_original = deepcopy(animik.positions)
        #positions_mean = animik.positions[0, 1:].reshape(1, 25, 3)
        positions_mean = np.mean(animik.positions[:, 1:], axis=0).reshape(1, 25, 3)
        positions_mean = np.repeat(positions_mean, len(targets), axis=0)
        animik.positions[:, 1:] = positions_mean
        ik = JacobianInverseKinematics(animik, targetmap, iterations=100, damping=2.0, silent=True, translate=False)
        ik()
    '''
    '''
    if dataset == 'ntu':
        positions = np.mean(animik.positions,axis=0)
        for j in range(len(targets)):
            animik.positions[j] = positions
    '''
    qs = animik.rotations.qs
    eulers = []
    for i in range(len(qs)):
        r = R.from_quat(qs[i])
        euler = r.as_euler('xyz', degrees=True)
        eulers.append(deepcopy(euler))
    eulers = np.array(eulers)
    eulers[:, :, 0] = 360 * (eulers[:, :, 0] < 0) + eulers[:, :, 0]
    #plus360 = eulers + 360
    #eulers = eulers + (360)*(plus360<=270)
    '''
    if dataset == 'ntu':
        return animik,eulers,positions_original
    else:
    '''
    return animik, eulers

def ad_ik(ik,targets,translate = False,num=100, dataset= 'empty'):
    initial = np.zeros((1, 3))
    targets = np.insert(targets, 0, targets[:, 0, :], axis=1)
    animik = ik.copy()
    animik.positions[:, 0, :] = targets[:, 0, :]
    targetmap = {}
    for ti in range(targets.shape[1]):
        targetmap[ti] = targets[:, ti]
    ik2 = JacobianInverseKinematics(animik, targetmap, iterations=num, damping=2.0, silent=True, translate=translate)
    ik2()
    qs = animik.rotations.qs
    eulers = []
    for i in range(len(qs)):
        r = R.from_quat(qs[i])
        euler = r.as_euler('xyz', degrees=True)
        eulers.append(deepcopy(euler))
    eulers = np.array(eulers)
    eulers[:, :, 0] = 360 * (eulers[:, :, 0] < 0) + eulers[:, :, 0]
    #plus360 = eulers + 360
    #eulers = eulers + (360)*(plus360<=270)
    return animik, eulers
def animation_euler(euler, anim):
    qs = []
    for i in range(len(anim.rotations.qs)):
        r = R.from_euler('xyz', euler[i], degrees=True)
        q = r.as_quat()
        qs.append(deepcopy(q))
    qs = np.array(qs)
    anim.rotations.qs = qs
    return anim

def correction_deg(euler, deg_out = False):
    '''
    deg_max2 = np.max(euler, axis=0)
    deg_min2 = np.min(euler, axis=0)
    deg_difference2 = deg_max2 - deg_min2
    deg_index = np.argwhere(deg_difference2 > 300)
    for i in range(len(deg_index)):
        euler[:, deg_index[i, 0], deg_index[i, 1]] = \
            (euler[:, deg_index[i, 0], deg_index[i, 1]] < 0) * 360 + euler[:,deg_index[i, 0], deg_index[i, 1]]
    deg_max = np.max(euler, axis=0)
    deg_min = np.min(euler, axis=0)
    deg_difference = deg_max - deg_min
    '''
    a = (euler[:,:,0]<0)
    euler[:,:,0] = 360*(euler[:,:,0]<0)+euler[:,:,0]
    max = np.max(euler,axis=0)
    min = np.min(euler,axis=0)
    if deg_out == True:
        return euler,max,min
    else:
        return euler


