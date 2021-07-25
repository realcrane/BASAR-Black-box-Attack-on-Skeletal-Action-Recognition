import os
import pickle
import numpy as np
from gekko import GEKKO
import shutil
from feeder.feeder import Feeder
from copy import deepcopy
import torch
import torch.nn as nn
import tools.tools as tools

class evaluation_ntu_single(object):
    def __init__(self,target_sample,adversarial_sample,dataset):
        if dataset == 'ntu':
            for num_frames in range(0, len(target_sample[0])):
                if np.all(target_sample[:, num_frames, :] == 0):
                    target_sample = target_sample[:,:num_frames,:]
                    adversarial_sample = adversarial_sample[:,:num_frames,:]
                    break
            self.target = target_sample
            self.ad = adversarial_sample
            neighbor_link = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                             (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                             (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                             (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                             (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = np.array(neighbor_link, dtype=int) - 1
            self.neighbor_link = neighbor_link
        elif dataset == 'hdm05':
            self.target = target_sample
            self.ad = adversarial_sample
            neighbor_link = [(1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
                             (7, 8), (8, 9), (9, 10), (11, 12), (12, 13),
                              (14, 15), (14, 16), (16, 17), (17, 18),
                             (18, 19), (19, 20), (14, 21), (21, 22), (22, 23),
                             (23, 24), (24, 25)]#(13, 14),(11, 1), (11, 6)
            neighbor_link = np.array(neighbor_link, dtype=int) - 1
            self.neighbor_link = neighbor_link
        elif dataset == 'kinetics':
            self.target = target_sample
            self.ad = adversarial_sample
            neighbor_link =[(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
             (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
             (16, 14)]
            neighbor_link = np.array(neighbor_link, dtype=int) - 1
            self.neighbor_link = neighbor_link
    def bone_length_difference(self):
        adboneVecs = self.ad[:, :, self.neighbor_link[:, 0]] - self.ad[:, :, self.neighbor_link[:, 1]]
        adboneLengths = np.sum(adboneVecs * adboneVecs, axis=0)
        adboneLengths = np.sqrt(adboneLengths)
        refboneVecs = self.target[:, :, self.neighbor_link[:, 0]] - self.target[:, :, self.neighbor_link[:, 1]]
        refboneLengths = np.sum(refboneVecs * refboneVecs, axis=0)
        refboneLengths = np.sqrt(refboneLengths)
        bone_diff = np.mean((np.abs(refboneLengths - adboneLengths)) / np.abs(refboneLengths))
        return bone_diff
    def speed_difference(self):
        tar_Sp = self.target[:, 1:, :] - self.target[:, :-1, :]
        ad_Sp = self.ad[:, 1:, :] - self.ad[:, :-1, :]
        #speed_diff = np.mean(tools.get_diff_np(tar_Sp.transpose(1, 2, 0), ad_Sp.transpose(1, 2, 0)))
        sp_diff = np.sum((tar_Sp - ad_Sp) ** 2, axis=0)
        sp_diff_mean = np.mean(np.sqrt(sp_diff))
        return sp_diff_mean
    def acc_difference(self):
        refAcc = (self.target[:, 2:, :] - 2 * self.target[:, 1:-1, :] + self.target[:, :-2, :])
        adAcc = (self.ad[:, 2:, :] - 2 * self.ad[:, 1:-1, :] + self.ad[:, :-2, :])
        #acc_diff = np.mean(tools.get_diff_np(refAcc.transpose(1, 2, 0), adAcc.transpose(1, 2, 0)))
        acc_diff = np.sum((refAcc - adAcc) ** 2, axis=0)
        acc_diff_mean = np.mean(np.sqrt(acc_diff))
        return acc_diff_mean
class evaluation_euler(object):
    def __init__(self, target_euler, adversarial_euler, dataset):
        self.target = target_euler
        self.ad = adversarial_euler
    def speed_difference(self):
        tar_Sp = self.target[1:, :, :] - self.target[:-1, :, :]
        ad_Sp = self.ad[1:, :, :] - self.ad[:-1, :, :]
        #speed_diff = np.mean(tools.get_diff_np(tar_Sp.transpose(1, 2, 0), ad_Sp.transpose(1, 2, 0)))
        sp_diff = (tar_Sp - ad_Sp) ** 2
        sp_diff_mean = np.mean(np.sqrt(sp_diff))
        return sp_diff_mean
    def acc_difference(self):
        refAcc = (self.target[2:, :, :] - 2 * self.target[1:-1, :, :] + self.target[:-2, :, :])
        adAcc = (self.ad[2:, :, :] - 2 * self.ad[1:-1, :, :] + self.ad[:-2, :, :])
        #acc_diff = np.mean(tools.get_diff_np(refAcc.transpose(1, 2, 0), adAcc.transpose(1, 2, 0)))
        acc_diff = (refAcc - adAcc) ** 2
        acc_diff_mean = np.mean(np.sqrt(acc_diff))
        return acc_diff_mean