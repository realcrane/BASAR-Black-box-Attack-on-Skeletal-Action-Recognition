import os
import pickle
import numpy as np
from gekko import GEKKO
import shutil
from feeder.feeder import Feeder
from copy import deepcopy
import torch
import tools.tools as tools
import torch.nn as nn

class bone_constraints(object):
    def __init__(self, adData, refData,dataset = 'self_define'):
        super().__init__()
        ndim = adData.ndim
        if dataset == 'ntu':
            neighbor_link = [(1, 2), (2,21),(3, 21), (4, 3), (5, 21),
                            (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                            (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                            (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                            (21, 2),(22, 23), (23, 8), (24, 25), (25, 12)]
            number_frames = 300
            for p in range(0, number_frames):
                if torch.all(refData[:, p, :] == 0):
                    break
        elif dataset == 'self_define':
            neighbor_link = [(1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
                              (7, 8), (8, 9), (9, 10), (11, 12), (12, 13),
                              (13, 14), (14, 15), (14, 16), (16, 17), (17, 18),
                              (18, 19), (19, 20), (14, 21),(21, 22), (22, 23),
                              (23, 24), (24, 25), (11, 1), (11, 6)]
            p = 60
        elif dataset == 'kinetics':
            neighbor_link =[(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
             (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
             (16, 14)]
            p = 300
        neighbor_link=torch.tensor(neighbor_link,dtype=int)-1


        self.neighbor_link = neighbor_link
        self.adData = adData
        self.refData = refData
        self.p = p
        self.ndim = ndim
    def bone_length(self):
        if self.ndim == 3:
            adboneVecs = self.adData[:, :, self.neighbor_link[:, 0]] - self.adData[:, :, self.neighbor_link[:, 1]]
            adboneLengths = torch.sum(adboneVecs*adboneVecs, axis=0)
            refboneVecs = self.refData[:, :, self.neighbor_link[:, 0]] - self.refData[:, :, self.neighbor_link[:, 1]]
            refboneLengths = torch.sum(refboneVecs*refboneVecs, axis=0)
            diff = (torch.abs(refboneLengths[0:self.p] - adboneLengths[0:self.p])) / torch.abs(refboneLengths[0:self.p])
            #length_diff = torch.mean(diff)
            return diff, adboneLengths, refboneLengths
        elif self.ndim == 4:
            adboneVecs = self.adData[:, :, :, self.neighbor_link[:, 0]] - self.adData[:, :, :, self.neighbor_link[:, 1]]
            adboneLengths = torch.sum(adboneVecs*adboneVecs, axis=1)
            refboneVecs = self.refData[:, :, :, self.neighbor_link[:, 0]] - self.refData[:, :, :, self.neighbor_link[:, 1]]
            refboneLengths = torch.sum(refboneVecs*refboneVecs, axis=1)
            diff = torch.abs(refboneLengths[:,0:self.p,:] - adboneLengths[:,0:self.p,:]) / torch.abs(refboneLengths[:,0:self.p,:])
            #length_diff = torch.mean(diff,axis=(1,2))
            return diff, adboneLengths, refboneLengths
    def acc(self):
        if self.ndim == 3:
            refAcc = (self.refData[:, 2:, :] - 2 * self.refData[:, 1:-1, :] + self.refData[:, :-2, :])
            adAcc = (self.adData[:, 2:, :] - 2 * self.adData[:, 1:-1, :] + self.adData[:, :-2, :])
            diff = torch.mean(tools.get_diff(refAcc.permute(1, 2, 0), adAcc.permute(1, 2, 0)))
            return diff,adAcc,refAcc
        elif self.ndim == 4:
            diffes = []
            refAcc = (self.refData[:, :, 2:, :] - 2 * self.refData[:, :, 1:-1, :] + self.refData[:, :, :-2, :])
            adAcc = (self.adData[:, :, 2:, :] - 2 * self.adData[:, :, 1:-1, :] + self.adData[:, :, :-2, :])
            for i in range(len(self.adData)):
                diff = torch.mean(tools.get_diff(refAcc[i].permute(1, 2, 0), adAcc[i].permute(1, 2, 0)))
                diffes.append(deepcopy(diff))
            diffes = torch.tensor(diffes)
            return diffes, adAcc, refAcc
    def l2_norm(self):
        if self.ndim == 3:
            diff = torch.mean(tools.get_diff(self.adData.permute(1, 2, 0), self.refData.permute(1, 2, 0)))
        elif self.ndim == 4:
            diffes = []
            for i in range(len(self.adData)):
                diff = torch.mean(tools.get_diff(self.adData[i].permute(1, 2, 0), self.refData[i].permute(1, 2, 0)))
                diffes.append(deepcopy(diff))
            diffes = torch.tensor(diffes)
            return diffes


class bone_constraints_np(object):
    def __init__(self, adData, refData,dataset):
        super().__init__()
        ndim = adData.ndim
        if dataset == 'ntu':
            neighbor_link = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                            (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                            (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                            (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                            (22, 23), (23, 8), (24, 25), (25, 12)]

        elif dataset == 'self_define':
            neighbor_link = [(1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
                             (7, 8), (8, 9), (9, 10), (11, 12), (12, 13),
                             (13, 14), (14, 15), (14, 16), (16, 17), (17, 18),
                             (18, 19), (19, 20), (14, 21), (21, 22), (22, 23),
                             (23, 24), (24, 25), (11, 1), (11, 6)]

        elif dataset == 'kinetics':
            neighbor_link =[(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
             (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
             (16, 14)]
        neighbor_link=np.array(neighbor_link,dtype=int)-1
        self.neighbor_link = neighbor_link
        self.adData = adData
        self.refData = refData
        self.ndim = ndim
    def bone_length(self):
        if self.ndim == 3:
            adboneVecs = self.adData[:, :, self.neighbor_link[:, 0]] - self.adData[:, :, self.neighbor_link[:, 1]]
            adboneLengths = np.sum(adboneVecs*adboneVecs, axis=0)
            refboneVecs = self.refData[:, :, self.neighbor_link[:, 0]] - self.refData[:, :, self.neighbor_link[:, 1]]
            refboneLengths = np.sum(refboneVecs*refboneVecs, axis=0)
            #diff = (np.abs(refboneLengths[0:self.p] - adboneLengths[0:self.p])) / np.abs(refboneLengths[0:self.p])
            #length_diff = np.mean(diff)
            return  adboneLengths, refboneLengths
        elif self.ndim == 4:
            adboneVecs = self.adData[:, :, :, self.neighbor_link[:, 0]] - self.adData[:, :, :, self.neighbor_link[:, 1]]
            adboneLengths = np.sum(adboneVecs*adboneVecs, axis=1)
            refboneVecs = self.refData[:, :, :, self.neighbor_link[:, 0]] - self.refData[:, :, :, self.neighbor_link[:, 1]]
            refboneLengths = np.sum(refboneVecs*refboneVecs, axis=1)
            #diff = np.abs(refboneLengths[:,0:self.p,:] - adboneLengths[:,0:self.p,:]) / np.abs(refboneLengths[:,0:self.p,:])
            #length_diff = np.mean(diff,axis=(1,2))
            return  adboneLengths, refboneLengths
    def acc(self):
        if self.ndim == 3:
            refAcc = (self.refData[:, 2:, :] - 2 * self.refData[:, 1:-1, :] + self.refData[:, :-2, :])
            adAcc = (self.adData[:, 2:, :] - 2 * self.adData[:, 1:-1, :] + self.adData[:, :-2, :])
            diff = np.mean(tools.get_diff_np(refAcc.transpose(1, 2, 0), adAcc.transpose(1, 2, 0)))
            return diff,adAcc,refAcc
        elif self.ndim == 4:
            diffes = []
            refAcc = (self.refData[:, :, 2:, :] - 2 * self.refData[:, :, 1:-1, :] + self.refData[:, :, :-2, :])
            adAcc = (self.adData[:, :, 2:, :] - 2 * self.adData[:, :, 1:-1, :] + self.adData[:, :, :-2, :])
            for i in range(len(self.adData)):
                diff = np.mean(tools.get_diff_np(refAcc[i].transpose(1, 2, 0), adAcc[i].transpose(1, 2, 0)))
                diffes.append(deepcopy(diff))
            diffes = np.array(diffes)
            return diffes, adAcc, refAcc
    def l2_norm(self):
        if self.ndim == 3:
            diff = np.mean(tools.get_diff_np(self.adData.transpose(1, 2, 0), self.refData.transpose(1, 2, 0)))
        elif self.ndim == 4:
            diffes = []
            for i in range(len(self.adData)):
                diff = np.mean(tools.get_diff_np(self.adData[i].transpose(1, 2, 0), self.refData[i].transpose(1, 2, 0)))
                diffes.append(deepcopy(diff))
            diffes = np.array(diffes)
            return diffes

