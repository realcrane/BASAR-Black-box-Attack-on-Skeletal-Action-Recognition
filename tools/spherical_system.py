import os
import pickle
import numpy as np
from copy import deepcopy
from tools.bone_constraints import bone_constraints
from tools.bone_constraints import bone_constraints_np
import tools.tools as tools
import torch
import torch.nn as nn

def spherical_coordinate(data,dataset):
    data_spherical = torch.zeros((2,60,24)).float().cuda()
    data1 = deepcopy(data)
    neighbor_link, r = tools.bone_length(data1, dataset)
    origin = data1[:, :, neighbor_link[:, 0]]
    joint = data1[:, :, neighbor_link[:, 1]]
    theta = torch.atan2(torch.sqrt((joint[0, :, :] - origin[0, :, :]) ** 2 + (joint[1, :, :] - origin[1, :, :]) ** 2),(joint[2, :, :] - origin[2, :, :]))
    phi = torch.atan2(joint[1, :, :] - origin[1, :, :], joint[0, :, :] - origin[0, :, :])
    data_spherical[0] = theta;data_spherical[1] = phi
    return data_spherical, r, origin

def cartesian_coordinate(data_spherical,r,dataset):
    p = len(data_spherical[0,:,0])
    if dataset == 'ntu':
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
    neighbor_link = torch.tensor(neighbor_link, dtype=int) - 1
    data = torch.zeros((3, p, 25)).float().cuda()
    theta = data_spherical[0]
    phi = data_spherical[1]
    for i in range (0,24):
        data[0,:,neighbor_link[i,1]] = data[0,:,neighbor_link[i,0]] + r[:,i]*torch.sin(theta[:,i])*torch.cos(phi[:, i])
        data[1, :, neighbor_link[i, 1]] = data[1,:,neighbor_link[i,0]] + r[:,i]*torch.sin(theta[:,i])*torch.sin(phi[:, i])
        data[2, :, neighbor_link[i, 1]] = data[2, :, neighbor_link[i, 0]] + r[:, i] * torch.cos(theta[:, i])

    return data.float().cuda()

def orthogonal_perturbation_spherical(delta, prev_sample, target_sample,dataset):
    number_frames = len(prev_sample[0,:,0])
    if dataset == 'self_define':
        i = 60
        # Generate perturbation
        perturb2 = torch.randn(2, number_frames, number_frames).cuda()
        b2 = tools.get_diff(perturb2, torch.zeros_like(perturb2).cuda())
        # b2 = np.maximum(b2, 0.00001)
        perturb2 = perturb2.permute(1, 2, 0)
        perturb2 /= b2
        perturb2 *= delta * torch.mean(tools.get_diff(target_sample, prev_sample))
        perturb2 = perturb2.permute(2, 0, 1)
        perturb = perturb2[:, :, 0:24]
        # Project perturbation onto sphere around target
        diff = target_sample - prev_sample
        diff = diff.permute(1, 2, 0)
        b3 = tools.get_diff(target_sample, prev_sample).float()
        diff /= b3
        diff = diff.permute(2, 0, 1)
    for i, channel in enumerate(diff):
        perturb[i] -= torch.mm(perturb2[i], channel) * channel
    return perturb