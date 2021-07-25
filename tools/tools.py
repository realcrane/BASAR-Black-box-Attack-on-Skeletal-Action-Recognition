import os
import pickle
import numpy as np
from gekko import GEKKO
import shutil
from feeder.feeder import Feeder
from copy import deepcopy
import torch
import torch.nn as nn

def bone_weights(target,ad_ori,datasets = 'ntu'):
    ad = deepcopy(ad_ori)
    if datasets == 'ntu':
        root_tar = target[:, :, 0]
        root_ad = ad[:, :, 0]
        tar1 = target[:, :, 1] - root_tar
        tar20 = target[:, :, 20] - root_tar
        tar2 = target[:, :, 2] - root_tar
        tar3 = target[:, :, 3] - root_tar
        ad[:, :, 1] = root_ad + tar1
        ad[ :, :, 20] = root_ad + tar20
        ad[ :, :, 2] = root_ad + tar2
        ad[ :, :, 3] = root_ad + tar3
    elif datasets == 'hdm05':
        ad[:,:,11:14] = target[:,:,11:14]
    c = ad.cpu().numpy()
    a = target.cpu().numpy()
    tar_length = bone_length_np(a, dataset='ntu_original')
    in_length = bone_length_np(c, dataset='ntu_original')
    return ad
def ntu_rebuild(data,roll_back = False):
    index_new = np.array([1,10,12,13,14,15,16,17,20,21,22,23,2,3,4,5,6,7,8,9,11,18,19,24,25])
    index_ori = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
    index_new = index_new - 1
    index_ori = index_ori - 1
    ndim = data.ndim
    data2 = np.zeros_like(data)
    if roll_back == False:
        if ndim == 3:
            data2[:,:,index_new] = data[:,:,index_ori]
        elif ndim == 4:
            data2[:,:, :, index_new] = data[:, :,:, index_ori]
    elif roll_back == True:
        if ndim == 3:
            data2[:,:,index_ori] = data[:,:,index_new]
        elif ndim == 4:
            data2[:,:, :, index_ori] = data[:, :,:, index_new]
    return data2
def bone_length_np(data,dataset = 'self_define',scale =False):
    if dataset == 'ntu_original':
        neighbor_link = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                         (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                         (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                         (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                         (22, 23), (23, 8), (24, 25), (25, 12)]

    elif dataset == 'self_define':
        neighbor_link = [(1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
                         (7, 8), (8, 9), (9, 10), (11, 12), (12, 13),
                         (13, 14), (14, 15), (14, 16),(16, 17), (17, 18),
                         (18, 19), (19, 20), (14, 21), (21, 22), (22, 23),
                         (23, 24), (24, 25), (11, 1), (11, 6)]
    elif dataset == 'ntu_rebuild':
        neighbor_link = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 6),
                             (6, 7), (7, 8), (8, 9), (1, 10), (10, 11),
                         (11, 12), (12, 13), (11, 14),(14, 15), (15, 16),
                         (16, 17), (17, 18), (18, 19), (11, 20), (20, 21),
                         (21, 22), (22, 23), (23, 24), (24, 25)]
    elif dataset == 'kinetics':
        neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
                     (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
                     (16, 14)]
        p = 300
    neighbor_link = np.array(neighbor_link, dtype=int) - 1
    ndim = data.ndim
    if ndim == 3:
        adboneVecs = data[:, :, neighbor_link[:, 0]] - data[:, :, neighbor_link[:, 1]]
        adboneLengths = np.sum(adboneVecs * adboneVecs, axis=0)
        if scale == True:
            offsets = np.mean(adboneVecs,axis =1)
            length = np.sum(offsets ** 2, axis=0)
            anim_scale = np.mean(offsets)
    elif ndim == 4:
        adboneVecs = data[:, :, :, neighbor_link[:, 0]] - data[:, :, :, neighbor_link[:, 1]]
        adboneLengths = np.sum(adboneVecs * adboneVecs, axis=1)
        if scale == True:
            offsets = np.mean(np.mean(adboneVecs,axis = 2),axis=0)
            anim_scale = np.mean(offsets)
    adboneLengths = np.sqrt(adboneLengths)
    if scale == True:
        return adboneLengths,offsets,anim_scale
    else:
        return adboneLengths
def bone_length(data,dataset = 'self_define'):
    if dataset == 'ntu':
        neighbor_link = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                         (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                         (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                         (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                         (22, 23), (23, 8), (24, 25), (25, 12)]
        number_frames = 300
        for p in range(0, number_frames):
            if torch.all(data[:, p, :] == 0):
                break
    elif dataset == 'self_define':
        neighbor_link = [(1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
                         (7, 8), (8, 9), (9, 10), (11, 12), (12, 13),
                         (13, 14), (14, 15), (14, 16),(16, 17), (17, 18),
                         (18, 19), (19, 20), (14, 21), (21, 22), (22, 23),
                         (23, 24), (24, 25), (11, 1), (11, 6)]
        p = 60
    elif dataset == 'kinetics':
        neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
                     (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
                     (16, 14)]
        p = 300
    neighbor_link = torch.tensor(neighbor_link, dtype=int) - 1
    ndim = data.ndim
    if ndim == 3:
        adboneVecs = data[:, :, neighbor_link[:, 0]] - data[:, :, neighbor_link[:, 1]]
        adboneLengths = torch.sum(adboneVecs * adboneVecs, axis=0)
    elif ndim == 4:
        adboneVecs = data[:, :, :, neighbor_link[:, 0]] - data[:, :, :, neighbor_link[:, 1]]
        adboneLengths = torch.sum(adboneVecs * adboneVecs, axis=1)
    adboneLengths = torch.sqrt(adboneLengths)
    return adboneLengths


def forward_perturbation(epsilon, prev_sample, target_sample):

    perturb = (target_sample - prev_sample)
    perturb = perturb.permute(1,2,0)
    #perturb = np.transpose(perturb, [1, 2, 0])  # (3,300,25), (300,25,3)
    b = get_diff(target_sample, prev_sample)
    perturb /= b
    perturb *= epsilon
    #perturb = np.transpose(perturb, [2, 0, 1])
    perturb = perturb.permute(2,0,1)
    return perturb
def forward_perturbation_sgn(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample)
    #perturb = np.transpose(perturb, [1, 2, 0])  # (3,300,25), (300,25,3)
    b = get_diff(target_sample.permute(1,0), prev_sample.permute(1,0))
    perturb /= b
    perturb *= epsilon
    #perturb = np.transpose(perturb, [2, 0, 1])
    return perturb
def forward_perturbation_ik_positions(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    #perturb = np.transpose(perturb, [1, 2, 0])  # (3,300,25), (300,25,3)
    b = get_diff_np(np.transpose(target_sample,[2,0,1]), np.transpose(prev_sample,[2,0,1]))
    b = np.maximum(b, 1e-10)
    perturb /= b
    perturb *= epsilon
    #perturb = np.transpose(perturb, [2, 0, 1])
    return perturb
def forward_perturbation_angle(epsilon, prev_sample, target_sample):
	perturb = (target_sample - prev_sample).astype(np.float32)
	perturb /= get_diff_angle(target_sample, prev_sample)
	perturb *= epsilon
	return perturb


def get_diff(sample_1, sample_2):
    # (3,25,300) (300,3,25)
    # sample_1 = np.transpose(sample_1, [2, 0, 1])
    # sample_2 = np.transpose(sample_2, [2, 0, 1])
    diff = []
    for i, channel in enumerate(sample_1):
        diff.append((torch.norm(channel - sample_2[i])))
    diff = torch.tensor(diff).cuda()
    return diff.float()

def get_diff_np(sample_1, sample_2,delete = False):
    # (3,25,300) (300,3,25)
    # sample_1 = np.transpose(sample_1, [2, 0, 1])
    # sample_2 = np.transpose(sample_2, [2, 0, 1])

    if delete == True:
        for i in range(0, len(sample_1)):
            if np.all(sample_1[i, :, :] == 0):
                sample_1 = sample_1[:i, :, :]
                sample_2 = sample_2[:i, :, :]
                break

    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)

def get_diff_angle(sample_1, sample_2):
    # (3,25,300) (300,3,25)
    sample_1 = np.transpose(sample_1, [2, 0, 1])
    sample_2 = np.transpose(sample_2, [2, 0, 1])
    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)

def orthogonal_perturbation_angle(delta, prev_euler, target_euler):
    number_frames = len(target_euler[:,0,0])
    # Generate perturbation
    perturb2 = np.random.randn(number_frames, number_frames,3)
    b2 = get_diff_angle(perturb2, np.zeros_like(perturb2))
    b2 = np.maximum(b2, 1e-10)
    perturb2 /= b2
    perturb2 *= delta * np.mean(get_diff_angle(target_euler, prev_euler))
    perturb = perturb2[:, 0:26, :]
    # Project perturbation onto sphere around target
    diff = (target_euler - prev_euler).astype(np.float32)
    b3 = get_diff_angle(target_euler, prev_euler)
    diff /= b3
    diff = np.transpose(diff, [2, 0, 1])
    perturb2 = np.transpose(perturb2, [2, 0, 1])
    perturb = np.transpose(perturb, [2, 0, 1])
    for i, channel in enumerate(diff):
        perturb[i] -= np.dot(perturb2[i], channel) * channel
    perturb = np.transpose(perturb, [1, 2, 0])
    # Check overflow and underflow
    overflow = (prev_euler+perturb) - 359.99
    perturb -= overflow * (overflow > 0)
    underflow = (prev_euler + perturb) + 179.99
    perturb -= underflow * (underflow < 0)
    return perturb

def orthogonal_perturbation(delta, prev_sample, target_sample, dataset = 'none'):
    number_frames = len(prev_sample[0,:,0])
    num_joints = len(prev_sample[0,0,:])
    # Generate perturbation
    perturb2 = torch.randn(3, number_frames, number_frames).cuda()
    #perturb2 = perturb2.cpu().numpy()
    '''
    if dataset == 'ntu':
        perturb2[:, :, 0] = 0
    '''
    if dataset == 'hdm05':
    #perturb2[:,:,0] = 0
        perturb2[:,:,5] = perturb2[:,:,0]
        perturb2[:, :,10] = perturb2[:, :, 0]
        perturb2[:,:,17] = perturb2[:,:,19]
        perturb2[:,:,22] = perturb2[:,:,24]
    b2 = get_diff(perturb2, torch.zeros_like(perturb2).cuda())
    #b2 = np.maximum(b2, 0.00001)
    perturb2 = perturb2.permute(1, 2, 0)
    perturb2 /= b2
    perturb2 *= delta * torch.mean(get_diff(target_sample, prev_sample))
    perturb2 = perturb2.permute(2, 0, 1)
    if num_joints <= number_frames:
        perturb = perturb2[:, :, 0:num_joints]
    else:
        perturb = torch.randn_like(target_sample)
        perturb = perturb.permute(1, 2, 0)
        perturb /= b2
        perturb = perturb.permute(2, 0, 1)
        perturb[:,:,:number_frames] = perturb2
    # Project perturbation onto sphere around target
    diff = target_sample - prev_sample
    diff = diff.permute(1, 2, 0)
    b3 = get_diff(target_sample, prev_sample).float()
    #b3 = np.maximum(b3, 0.00001)
    diff /= b3
    diff = diff.permute(2, 0, 1)
    # diff = np.transpose(diff, [2, 0, 1])
    # perturb2 = np.transpose(perturb2, [2, 0, 1])
    # perturb = np.transpose(perturb, [2, 0, 1])

    for i, channel in enumerate(diff):
        perturb[i] -= torch.mm(perturb2[i], channel) * channel
    # perturb = np.transpose(perturb, [1, 2, 0])
    #perturb_np = perturb.cpu().numpy()
    return perturb
def orthogonal_perturbation_sgn(delta, prev_sample, target_sample, dataset = 'none'):
    number_frames = len(prev_sample[:,0])
    num_joints = len(prev_sample[0,:])
    if num_joints >= number_frames:
        # Generate perturbation
        perturb2 = torch.randn(75, 75).cuda()
        #perturb2 = perturb2.cpu().numpy()
        '''
        if dataset == 'ntu':
            perturb2[:, :, 0] = 0
        '''
        if dataset == 'hdm05':
        #perturb2[:,:,0] = 0
            perturb2[:,:,5] = perturb2[:,:,0]
            perturb2[:, :,10] = perturb2[:, :, 0]
            perturb2[:,:,17] = perturb2[:,:,19]
            perturb2[:,:,22] = perturb2[:,:,24]
        b2 = get_diff(perturb2, torch.zeros_like(perturb2).cuda())
        # b2 = np.maximum(b2, 0.00001)
        perturb2 /= b2
        perturb2 *= delta * torch.mean(get_diff(target_sample.permute(1,0), prev_sample.permute(1,0)))
        perturb = perturb2[:,:number_frames]
        # Project perturbation onto sphere around target
        diff = target_sample - prev_sample
        b3 = get_diff(target_sample.permute(1,0), prev_sample.permute(1,0)).float()
        diff /= b3
        diff = diff.permute(1,0)
        # diff = np.transpose(diff, [2, 0, 1])
        # perturb2 = np.transpose(perturb2, [2, 0, 1])
        # perturb = np.transpose(perturb, [2, 0, 1])

        perturb -= torch.mm(perturb2, diff) * diff
        # perturb = np.transpose(perturb, [1, 2, 0])
        #perturb_np = perturb.cpu().numpy()

        return perturb.permute(1,0)



def draw_frames(sample_data):
    number_frames = len(sample_data[0,0,:,0,0])
    for p in range(0, number_frames):
        if np.all(sample_data[:,:,p,:,:] == 0):
            p = p -1
            break
    mul = 3
    p_int = int(p/mul)
    draw_samples_data = np.zeros((1,3,number_frames,25,1))
    draw_samples = draw_samples_data[0,:,:,:,0]
    sample = sample_data[0,:,:,:,0]
    draw_samples[:,0,:] = sample[:,0,:]
    draw_samples[:, p_int*mul+1:, :] = sample[:, p_int*mul+1:, :]
    for i in range (1,p_int+1):
        draw_samples[:,mul*i,:] = sample[:,mul*i,:]
        diff = (draw_samples[:,mul*i,:]-draw_samples[:,mul*(i-1),:])/3
        for j in range(1,mul):
            draw_samples[:,mul*(i-1) +j,:] = draw_samples[:,mul*(i-1),:] + j*diff
    return draw_samples_data

def postmatlab(orMotions,motions,datatype):
    if datatype == 'hdm05':
        core = np.load('../dataprecess/core/hdm05_core.npy')
    elif datatype == 'mhad':
        core = np.load('../dataprecess/core/mhad_core.npy')
    elif datatype == 'ntu':
        core = np.load('../dataprecess/core/ntu_core.npy')
    std = core[0]
    mean = core[1]
    for i in range(len(motions)):
        motions[i] = motions[i] * std + mean
        orMotions[i] = orMotions[i] * std + mean
    return orMotions, motions

def postmatlab_one(motions,datatype):
    if datatype == 'hdm05':
        core = np.load('../dataprecess/core/hdm05_core.npy')
    elif datatype == 'mhad':
        core = np.load('../dataprecess/core/mhad_core.npy')
    elif datatype == 'ntu':
        core = np.load('../dataprecess/core/ntu_core.npy')
    std = core[0]
    mean = core[1]
    for i in range(len(motions)):
        motions[i] = motions[i] * std + mean
    return  motions

def matlab_transpose(data,joints=75):
    ndim = data.ndim
    if ndim == 4:
        num_frames = len(data[0,0,:,0])
        data = np.transpose(data, [0, 2, 1, 3])
        matlab_data = data.reshape((-1, num_frames, joints), order='F')
    elif ndim == 3:
        num_frames = len(data[0,:,0])
        data = np.transpose(data, [1, 0, 2])
        matlab_data = data.reshape(( num_frames, joints), order='F')
    return matlab_data

def postprocessing_from_matlab(data):
    ndim = data.ndim
    if ndim == 3:
        num_frames = len(data[0,:,0])
        data = data.reshape((-1,num_frames,3,25), order = 'F')
        data = np.transpose(data, [0,2,1,3])
    elif ndim == 2:
        num_frames = len(data[:,0])
        data = data.reshape((num_frames,3,25), order = 'F')
        data = np.transpose(data, [1,0,2])
    return data
def postprocessing(motion, core, transpose = False):
    #data = np.transpose(data, [])
    std = core[0]
    mean = core[1]
    motion = np.transpose(motion,[1,2,0])
    motion = motion * std + mean
    if transpose == True:
        motion = np.transpose(motion,[2,0,1])
    return motion

def normalize(motion, core):
    #(60,25,3) (3,60,25)
    std = core[0]
    mean = core[1]
    motion = (motion - mean)/std
    motion = np.transpose(motion,[2,0,1])
    return motion

def ground(target_sample,ad_sample_original):
    ad_sample = deepcopy(ad_sample_original)
    sp_tar = target_sample[:, 1:, [3, 4, 8, 9]] - target_sample[:, :-1, [3, 4, 8, 9]]
    sp_length = np.sum(sp_tar * sp_tar, 0)
    sp_foot = np.mean(sp_length[:, [0, 1]], -1) + np.mean(sp_length[:, [2, 3]], -1)
    for i in range(len(sp_length)):
        if sp_foot[i] < 1e-3:
            ad_sample[1, i + 1, [3, 4, 8, 9]] = target_sample[1, i + 1, [3, 4, 8, 9]]
            if i == 0:
                ad_sample[1, 0, [3, 4, 8, 9]] = target_sample[1, 0, [3, 4, 8, 9]]
        else:
            foot1 = np.mean(target_sample[1, i + 1, [3, 4]])
            foot2 = np.mean(target_sample[1, i + 1, [8, 9]])
            if foot1 <= foot2:
                ad_sample[1, i + 1, [3, 4]] = target_sample[1, i + 1, [3, 4]]
                if i == 0:
                    ad_sample[1, 0, [3, 4]] = target_sample[1, 0, [3, 4]]
            else:
                ad_sample[1, i + 1, [8, 9]] = target_sample[1, i + 1, [8, 9]]
                if i == 0:
                    ad_sample[1, 0, [8, 9]] = target_sample[1, 0, [8, 9]]
    return ad_sample