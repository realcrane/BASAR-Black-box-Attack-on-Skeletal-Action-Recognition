import os
import pickle
import numpy as np
import gc
from gekko import GEKKO
import sys
sys.path.append('../motion')
sys.path.append('../')
from Quaternions import Quaternions
import BVH as BVH
import Animation as Animation
import tools.tools as tools
from copy import deepcopy
import tools.kinematics as kinematics
from tools.bone_constraints import bone_constraints_np
from motion.InverseKinematics import JacobianInverseKinematics
import shutil
from feeder.feeder import Feeder
from copy import deepcopy
from tools.bone_constraints import bone_constraints_np
import tools.tools as tools
import torch
import torch.nn as nn
import tools.evaluation as ev

def op_angle_ntu(target_sample_euler,adv_sample_euler,adv_sample_ik,weights_acc,length_constraints=True,out_euler = False):
    bound = np.load('../hdm05_bound.npy')
    bound = np.transpose(bound,[1,0,2])
    num_frames = len(target_sample_euler)
    target_sample_euler, deg_max, deg_min = kinematics.correction_deg(target_sample_euler, deg_out=True)
    adv_sample_euler, deg_max_ad, deg_min_ad = kinematics.correction_deg(adv_sample_euler, deg_out=True)
    if length_constraints == True:
        deg_min_copy = deepcopy(deg_min)
        deg_max_copy = deepcopy(deg_max)
        deg_min[deg_min > 2] = deg_min[deg_min > 2] - 2
        deg_min[deg_min < 0] = deg_min[deg_min < 0] - 2
        deg_max[deg_max < 358] = deg_max[deg_max < 358] + 2
        bound[10:13, :, 0] = deg_min_copy[10:13]
        bound[10:13, :, 1] = deg_max_copy[10:13]
    else:
        bound[:, :, 0] = deg_min_ad
        bound[:, :, 1] = deg_max_ad
    '''
    deg_min_copy = deepcopy(deg_min)
    deg_max_copy = deepcopy(deg_max)
    deg_min[deg_min >2] = deg_min[deg_min >2]-2
    deg_min[deg_min < 0] = deg_min[deg_min <0 ] - 2
    deg_max[deg_max<358] = deg_max[deg_max<358] + 2
    bound[:, :, 0] = deg_min
    bound[:, :, 1] = deg_max
    bound[10:13, :, 0] = deg_min_copy[10:13]
    bound[10:13, :, 1] = deg_max_copy[10:13]
    if joint_limits == 'strict':
        bound[:, :, 0] = deg_min
        bound[:, :, 1] = deg_max
    else:
        bound[11:15,:,0] = deg_min[11:15]
        bound[11:15,:,1] = deg_max[11:15]
    '''
    op_euler = deepcopy(adv_sample_euler)
    m = GEKKO()
    m.options.OTOL = 1.0e-4
    m.solver_options = ['max_iter 500', 'max_cpu_time 10000']
    m.options.SOLVER = 3
    m.options.IMODE = 3
    X = np.zeros_like((target_sample_euler), dtype=np.object)
    initial_val = 1
    for i in range(0, num_frames):
        for j in range(0, 26):
            for k in range(0, 3):
                if initial_val == 0:
                    X[i, j, k] = m.Var()
                elif initial_val == 1:
                    X[i, j, k] = m.Var(value=adv_sample_euler[i, j, k], lb=bound[j, k, 0], ub=bound[j, k, 1])
    XAcc = (X[2:, :, :] - 2 * X[1:-1, :, :] + X[:-2, :, :])
    # define speed and acc:
    tar_dec_Sp = target_sample_euler[1:, :, :] - target_sample_euler[:-1, :, :]
    ad_dec_Sp = adv_sample_euler[1:, :, :] - adv_sample_euler[:-1, :, :]
    tar_dec_Acc = (target_sample_euler[2:, :, :] - 2 * target_sample_euler[1:-1, :, :] + target_sample_euler[:-2, :, :])
    adv_dec_Acc = (adv_sample_euler[2:, :, :] - 2 * adv_sample_euler[1:-1, :, :] + adv_sample_euler[:-2, :, :])
    OBJ_ACC = np.sum(np.sum((tar_dec_Acc - XAcc) ** 2, axis=-1), axis=-1)
    #OBJ_Ad_ACC = np.sum(np.sum((adv_dec_Acc - XAcc) ** 2, axis=-1), axis=-1)
    OBJ_sample = np.sum(np.sum((X - adv_sample_euler) ** 2, axis=-1), axis=-1)
    #OBJ_Target = np.sum(np.sum((X - target_sample_euler) ** 2, axis=-1), axis=-1)

    for i in range(0, num_frames):
        m.Obj(OBJ_sample[i])
        if weights_acc > 0:
            if i < num_frames-2:
                m.Obj(weights_acc * OBJ_ACC[i])
    m.solve()
    if m.options.APPSTATUS == 1:
        for i in range(0, num_frames):
            for j in range(0, 26):
                for k in range(0, 3):
                    op_euler[i, j, k] = np.array(X[i, j, k].value)
    op_sample_ik = kinematics.animation_euler(op_euler, adv_sample_ik)
    op_max = np.max(op_euler, axis=0)
    op_min = np.min(op_euler, axis=0)
    op_dec_Sp = op_euler[1:, :, :] - op_euler[:-1, :, :]
    op_global = Animation.positions_global(op_sample_ik).astype(np.float32)
    op_sample = tools.ntu_rebuild(np.transpose(op_global[:, 1:, :], [2, 0, 1]), roll_back=True)
    if out_euler == True:
        return torch.from_numpy(op_sample).cuda(), op_global,op_euler
    else:
        return torch.from_numpy(op_sample).cuda(), op_global
def op_angle(target_sample_euler,adv_sample_euler,adv_sample_ik,core,weights_acc,straetgy,joint_limits = 'strict'):
    bound = np.load('../hdm05_bound.npy')
    bound = np.transpose(bound,[1,0,2])
    target_sample_euler, deg_max, deg_min = kinematics.correction_deg(target_sample_euler, deg_out=True)
    adv_sample_euler, deg_max_ad, deg_min_ad = kinematics.correction_deg(adv_sample_euler, deg_out=True)
    deg_min[deg_min >4] = deg_min[deg_min >4]-2
    deg_min[deg_min < 0] = deg_min[deg_min <0 ] - 2
    deg_max[deg_max<356] = deg_max[deg_max<356] + 2
    bound[11:15, :, 0] = deg_min[11:15]
    bound[11:15, :, 1] = deg_max[11:15]
    '''
    if joint_limits == 'strict':
        bound[:, :, 0] = deg_min
        bound[:, :, 1] = deg_max
    else:
        bound[11:15,:,0] = deg_min[11:15]
        bound[11:15,:,1] = deg_max[11:15]
    '''
    op_euler = deepcopy(adv_sample_euler)
    m = GEKKO()
    m.solver_options = ['max_iter 500', 'max_cpu_time 40000']
    m.options.SOLVER = 3
    m.options.IMODE = 3
    X = np.zeros_like((target_sample_euler), dtype=np.object)
    initial_val = 1
    for i in range(0, 60):
        for j in range(0, 26):
            for k in range(0, 3):
                if initial_val == 0:
                    X[i, j, k] = m.Var()
                elif initial_val == 1:
                    X[i, j, k] = m.Var(value=target_sample_euler[i, j, k], lb=bound[j, k, 0], ub=bound[j, k, 1])
    XAcc = (X[2:, :, :] - 2 * X[1:-1, :, :] + X[:-2, :, :])
    # define speed and acc:
    tar_dec_Sp = target_sample_euler[1:, :, :] - target_sample_euler[:-1, :, :]
    ad_dec_Sp = adv_sample_euler[1:, :, :] - adv_sample_euler[:-1, :, :]
    tar_dec_Acc = (target_sample_euler[2:, :, :] - 2 * target_sample_euler[1:-1, :, :] + target_sample_euler[:-2, :, :])
    adv_dec_Acc = (adv_sample_euler[2:, :, :] - 2 * adv_sample_euler[1:-1, :, :] + adv_sample_euler[:-2, :, :])
    OBJ_ACC = np.sum(np.sum((tar_dec_Acc - XAcc) ** 2, axis=-1), axis=-1)
    OBJ_Ad_ACC = np.sum(np.sum((adv_dec_Acc - XAcc) ** 2, axis=-1), axis=-1)
    OBJ_sample = np.sum(np.sum((X - adv_sample_euler) ** 2, axis=-1), axis=-1)
    OBJ_Target = np.sum(np.sum((X - target_sample_euler) ** 2, axis=-1), axis=-1)

    for i in range(0, 60):
        if straetgy == 2:
            m.Obj(OBJ_sample[i])
        elif straetgy == 3:
            m.Obj(OBJ_Target[i])
        if weights_acc > 0:
            if i < 58:
                if straetgy == 2:
                    m.Obj(weights_acc * OBJ_ACC[i])
                elif straetgy == 3:
                    m.Obj(weights_acc * OBJ_Ad_ACC[i])
    m.solve()
    if m.options.APPSTATUS == 1:
        for i in range(0, 60):
            for j in range(0, 26):
                for k in range(0, 3):
                    op_euler[i, j, k] = np.array(X[i, j, k].value)
    op_sample_ik = kinematics.animation_euler(op_euler, adv_sample_ik)
    op_max = np.max(op_euler, axis=0)
    op_min = np.min(op_euler, axis=0)
    op_dec_Sp = op_euler[1:, :, :] - op_euler[:-1, :, :]
    op_global = Animation.positions_global(op_sample_ik).astype(np.float32)
    op_sample = tools.normalize(op_global[:, 1:, :], core)
    return torch.from_numpy(op_sample).cuda(), op_global
def op_position(adversarial_sample,target_sample,dataset,weights_acc):
    diff = np.mean(tools.get_diff_np(np.transpose(adversarial_sample, [1, 2, 0]), np.transpose(target_sample, [1, 2, 0])))
    if diff >= 1:
        weight = 0.04
    elif 0.1 <= diff < 1:
        weight = 0.02
    else:
        weight = 0.01
    number_frames = len(target_sample[0,:,0])
    bone_sample = deepcopy(adversarial_sample)
    '''
    for p in range(0, number_frames):
        if np.all(adversarial_sample[:,p,:] == 0):
            break
        if p == number_frames - 1:
            p+= 1
            break
    adversarial_sample = adversarial_sample[:,0:p,:]
    '''
    X = np.zeros_like(target_sample, dtype=np.object)
    m = GEKKO()
    m.solver_options = ['max_iter 1000', 'max_cpu_time 20000']
    m.options.SOLVER = 3
    m.options.IMODE = 3
    initial_val = 1
    for i in range(0,number_frames):
        for j in range(0,18):
            for k in range(0,3):
                if initial_val == 0:
                    X[k, i, j] = m.Var()
                elif  initial_val == 1:
                    X[k, i, j] = m.Var(value=target_sample[k, i, j])
    OBJ_BONE = np.sum(np.sum((X - adversarial_sample)**2,axis=0),axis=-1)
    OBJ_ROOT = np.sum((X[:,0,:] - target_sample[:,0,:])**2)
    #objective = m.sum((X - adversarial_sample[:,0:p,:])**2)
    Bone_X_tar = bone_constraints_np(X,target_sample,dataset)
    X_bonelength,tar_bonelength = Bone_X_tar.bone_length()
    Bone_ad_tar = bone_constraints_np(adversarial_sample,target_sample,dataset)
    ad_bonelength,_ = Bone_ad_tar.bone_length()
    XAcc = (X[:, 2:, :] - 2 * X[:, 1:-1, :] + X[:, :-2, :])
    adACC = (adversarial_sample[:, 2:, :] - 2 * adversarial_sample[:, 1:-1, :] + adversarial_sample[:, :-2, :])
    tarAcc = (target_sample[:, 2:, :] - 2 * target_sample[:, 1:-1, :] + target_sample[:, :-2, :])
    OBJ_ACC = np.sum(np.sum((XAcc - tarAcc)**2,axis=0),axis=-1)

    for i in range(0,number_frames):
        for j in range(0,17):
            m.Equation(X_bonelength[i, j] <= tar_bonelength[i, j] * (1+weight))
            m.Equation(X_bonelength[i, j] >= tar_bonelength[i, j] * (1 - weight))
        m.Obj(OBJ_BONE[i])
        if i < number_frames-2:
            m.Obj(weights_acc*OBJ_ACC[i])
    m.solve(disp = False)
    if m.options.APPSTATUS == 1:
        for i in range(0, number_frames):
            for j in range(0, 18):
                for k in range(0, 3):
                    bone_sample[k,i,j] = np.array(X[k, i, j].value)
    '''
    Bone_X_ad = bone_constraints_np(bone_sample,adversarial_sample,dataset)
    X_bonelength,ad_bonelength = Bone_X_ad.bone_length()
    OpAcc = (bone_sample[:, 2:, :] - 2 * bone_sample[:, 1:-1, :] + bone_sample[:, :-2, :])
    evaluation_ad = ev.evaluation_ntu_single(target_sample, adversarial_sample,dataset)
    ad_speed_diff_ad = evaluation_ad.speed_difference()
    ad_acc_diff_ad = evaluation_ad.acc_difference()
    evaluation_op = ev.evaluation_ntu_single(target_sample, bone_sample,dataset)
    op_speed_diff = evaluation_op.speed_difference()
    op_acc_diff = evaluation_op.acc_difference()
    '''
    del X
    gc.collect()
    return torch.from_numpy(bone_sample).cuda()
def bone_con(adversarial_sample,target_sample, initial_val,dataset):
    number_frames = len(target_sample[0,:,0])
    bone_sample = deepcopy(adversarial_sample)
    '''
    for p in range(0, number_frames):
        if np.all(adversarial_sample[:,p,:] == 0):
            break
        if p == number_frames - 1:
            p+= 1
            break
    adversarial_sample = adversarial_sample[:,0:p,:]
    '''
    X = np.zeros((3, number_frames, 25), dtype=np.object)
    m = GEKKO()
    m.solver_options = ['max_iter 200', 'max_cpu_time 200']
    m.options.SOLVER = 3
    m.options.IMODE = 3
    for i in range(0,number_frames):
        for j in range(0,25):
            if j == 5 or j == 10 :
                continue
            else:
                for k in range(0,3):
                    if initial_val == 0:
                        X[k, i, j] = m.Var()
                    elif  initial_val == 1:
                        X[k, i, j] = m.Var(value=adversarial_sample[k, i, j])
    X[:, :, 5] = X[:, :, 0]
    X[:, :, 10] = X[:, :, 0]
                #X[k, i, j] = m.Var()
    OBJ_BONE = np.sum(np.sum((X - adversarial_sample)**2,axis=0),axis=-1)
    OBJ_ROOT = np.sum((X[:,0,:] - target_sample[:,0,:])**2)
    #objective = m.sum((X - adversarial_sample[:,0:p,:])**2)
    Bone_X_tar = bone_constraints_np(X,target_sample[:,0:p,:],dataset)
    X_bonelength,tar_bonelength = Bone_X_tar.bone_length()
    Bone_ad_tar = bone_constraints_np(adversarial_sample[:,0:p,:],target_sample[:,0:p,:],dataset)
    ad_bonelength,_ = Bone_ad_tar.bone_length()
    XAcc = (X[:, 2:, :] - 2 * X[:, 1:-1, :] + X[:, :-2, :])
    adACC = (adversarial_sample[:, 2:, :] - 2 * adversarial_sample[:, 1:-1, :] + adversarial_sample[:, :-2, :])
    tarAcc = (target_sample[:, 2:, :] - 2 * target_sample[:, 1:-1, :] + target_sample[:, :-2, :])
    OBJ_ACC = np.sum(np.sum((XAcc - tarAcc)**2,axis=0),axis=-1)
    weights_acc = 1
    weights_root = 0
    #m.Obj(objective)
    m.Obj(weights_root*OBJ_ROOT)
    for i in range(0,p):
        for j in range(0,24):
            m.Equation(X_bonelength[i,j] == tar_bonelength[i,j])
        m.Obj(OBJ_BONE[i])
        if i < p-2:
            m.Obj(weights_acc*OBJ_ACC[i])
    m.solve()
    if m.options.APPSTATUS == 1:
        for i in range(0, p):
            for j in range(0, 25):
                for k in range(0, 3):
                    bone_sample[k,i,j] = np.array(X[k, i, j].value)
    '''
    Bone_X_tar = bone_constraints_np(bone_sample,target_sample,dataset)
    diff_x_tar,X_bonelength,tar_bonelength = Bone_X_tar.bone_length()
    Bone_X_ad = bone_constraints_np(bone_sample,adversarial_sample,dataset)
    diff_ad_tar,X_bonelength,ad_bonelength = Bone_X_ad.bone_length()
    '''
    return bone_sample
def bone_con_root(adversarial_sample,target_sample, initial_val,dataset):
    number_frames = len(target_sample[0,:,0])
    bone_sample = deepcopy(adversarial_sample)
    for p in range(0, number_frames):
        if np.all(adversarial_sample[:,p,:] == 0):
            break
        if p == number_frames - 1:
            p+= 1
            break
    adversarial_sample = adversarial_sample[:,0:p,:]
    X = np.zeros((3, p, 25), dtype=np.object)
    m = GEKKO()
    m.solver_options = ['max_iter 200', 'max_cpu_time 200']
    m.options.SOLVER = 3
    m.options.IMODE = 3
    for i in range(0,p):
        for j in range(0,25):
            if j == 0 or j == 5 or j == 10 :
                continue
            else:
                for k in range(0,3):
                    if initial_val == 0:
                        X[k, i, j] = m.Var()
                    elif  initial_val == 1:
                        X[k, i, j] = m.Var(value=adversarial_sample[k, i, j])
    X[:,:,0] = X[:,:,0].astype(np.float32)
    X[:, :, 5] = X[:, :, 0]
    X[:, :, 10] = X[:, :, 0]
                #X[k, i, j] = m.Var()
    OBJ_BONE = np.sum(np.sum((X - adversarial_sample)**2,axis=0),axis=-1)
    OBJ_ROOT = np.sum((X[:,0,:] - target_sample[:,0,:])**2)
    #objective = m.sum((X - adversarial_sample[:,0:p,:])**2)
    Bone_X_tar = bone_constraints_np(X,target_sample[:,0:p,:],dataset)
    X_bonelength,tar_bonelength = Bone_X_tar.bone_length()
    Bone_ad_tar = bone_constraints_np(adversarial_sample[:,0:p,:],target_sample[:,0:p,:],dataset)
    ad_bonelength,_ = Bone_ad_tar.bone_length()
    XAcc = (X[:, 2:, :] - 2 * X[:, 1:-1, :] + X[:, :-2, :])
    adACC = (adversarial_sample[:, 2:, :] - 2 * adversarial_sample[:, 1:-1, :] + adversarial_sample[:, :-2, :])
    tarAcc = (target_sample[:, 2:, :] - 2 * target_sample[:, 1:-1, :] + target_sample[:, :-2, :])
    OBJ_ACC = np.sum(np.sum((XAcc - tarAcc)**2,axis=0),axis=-1)
    weights_acc = 1
    weights_root = 0
    #m.Obj(objective)
    m.Obj(weights_root*OBJ_ROOT)
    for i in range(0,p):
        for j in range(0,24):
            if j == 22 or j == 23 :
                continue
            else:
                m.Equation(X_bonelength[i,j] == tar_bonelength[i,j])
        m.Obj(OBJ_BONE[i])
        if i < p-2:
            m.Obj(weights_acc*OBJ_ACC[i])
    m.solve()
    if m.options.APPSTATUS == 1:
        for i in range(0, p):
            for j in range(0, 25):
                if j == 0 or j == 5 or j == 10:
                    bone_sample[:,:,j] = X[:,:,j]
                else:
                    for k in range(0, 3):
                        bone_sample[k,i,j] = np.array(X[k, i, j].value)

    '''
    Bone_X_tar = bone_constraints_np(bone_sample,target_sample,dataset)
    diff_x_tar,X_bonelength,tar_bonelength = Bone_X_tar.bone_length()
    Bone_X_ad = bone_constraints_np(bone_sample,adversarial_sample,dataset)
    diff_ad_tar,X_bonelength,ad_bonelength = Bone_X_ad.bone_length()
    '''
    return bone_sample,m.options.APPSTATUS

