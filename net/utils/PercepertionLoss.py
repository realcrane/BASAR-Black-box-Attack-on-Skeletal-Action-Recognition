import torch.nn as nn
import torch
import numpy as np


class PercepertionLoss(nn.Module):
    def __init__(self, perpLossType, classWeight, reconWeight, boneLenWeight):
        super().__init__()
        neighbor_link = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                        (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                        (22, 23), (23, 8), (24, 25), (25, 12)]

        jointWeights = [[[0.02, 0.02, 0.02, 0.02, 0.02,
                          0.02, 0.02, 0.02, 0.02, 0.02,
                          0.04, 0.04, 0.04, 0.04, 0.04,
                          0.02, 0.02, 0.02, 0.02, 0.02,
                          0.02, 0.02, 0.02, 0.02, 0.02]]]

        neighbor_link=torch.tensor(neighbor_link,dtype=int)-1
        jointWeights = torch.tensor(jointWeights, dtype=torch.float32)
        self.register_buffer('neighbor_link', neighbor_link)
        self.register_buffer('jointWeights', jointWeights)

        self.deltaT = 1 / 30
        self.perpLossType = perpLossType
        self.classWeight = classWeight
        self.reconWeight = reconWeight
        self.boneLenWeight = boneLenWeight

    def forward(self, refData, adData):

        N, C, T, V, M = refData.size()
        refData = refData.permute(0, 4, 2, 3, 1).contiguous()
        refData = refData.view(N * M, T, V, C)

        adData = adData.permute(0, 4, 2, 3, 1).contiguous()
        adData = adData.view(N * M, T, V, C)

        diff = adData - refData
        squaredLoss = torch.sum(torch.mul(diff, diff), dim=-1)
        #weightedSquaredLoss = squaredLoss * self.jointWeights
        squareCost = torch.sum(torch.sum(squaredLoss, axis=-1), axis=-1)
        #squareCost = torch.sum(torch.sum(weightedSquaredLoss, axis=-1), axis=-1)
        oloss = torch.mean(squareCost, axis=-1)

        if self.perpLossType == 'l2':

            return oloss

        elif self.perpLossType == 'acc':

            refAcc = (refData[:, 2:, :, : ] - 2 * refData[:, 1:-1, :, :] + refData[:, :-2, :, :])/self.deltaT/self.deltaT
            adAcc = (adData[:, 2:, :, :] - 2 * adData[:, 1:-1, :, :] + adData[:, :-2, :, :])/self.deltaT/self.deltaT

            diff = adAcc-refAcc
            jointAcc = torch.mean(torch.sum(torch.sum(torch.sum(torch.mul(diff, diff), axis=-1), axis=-1), axis=-1), axis=-1)

            return jointAcc * (1 - self.reconWeight) + oloss * self.reconWeight

        elif self.perpLossType == 'smoothness':

            adAcc = (adData[:, 2:, :, :] - 2 * adData[:, 1:-1, :, :] + adData[:, :-2, :, :]) / self.deltaT / self.deltaT

            jointAcc = torch.mean(torch.sum(torch.sum(torch.sum(torch.mul(adAcc, adAcc), axis=-1), axis=-1), axis=-1), axis=-1)

            return jointAcc * (1 - self.reconWeight) + oloss * self.reconWeight

        elif self.perpLossType == 'jerkness':

            adJerk = (adData[:, 3:, :, :] - 3 * adData[:, 2:-1, :, :] + 3 * adData[:, 1:-2, :, :] + adData[:, :-3, :, :])/self.deltaT/self.deltaT/self.deltaT

            jointJerk = torch.mean(torch.sum(torch.sum(torch.sum(torch.mul(adJerk, adJerk), axis=-1), axis=-1), axis=-1), axis=-1)

            return jointJerk * (1 - self.reconWeight) + oloss * self.reconWeight

        elif self.perpLossType == 'acc-jerk':

            refAcc = (refData[:, 2:, :, :] - 2 * refData[:, 1:-1, :, :] + refData[:, :-2, :, :]) / self.deltaT / self.deltaT

            adAcc = (adData[:, 2:, :, :] - 2 * adData[:, 1:-1, :, :] + adData[:, :-2, :, :]) / self.deltaT / self.deltaT

            diff = adAcc - refAcc
            jointAcc = torch.mean(torch.sum(torch.sum(torch.sum(torch.mul(diff, diff), axis=-1), axis=-1), axis=-1), axis=-1)

            adJerk = (adData[:, 3:, :, :] - 3 * adData[:, 2:-1, :, :] + 3 * adData[:, 1:-2, :, :] + adData[:, :-3, :, :]) / self.deltaT / self.deltaT / self.deltaT

            jointJerk = torch.mean(torch.sum(torch.sum(torch.sum(torch.mul(adJerk, adJerk), axis=-1), axis=-1), axis=-1), axis=-1)

            jerkWeight = 0.7

            return jointJerk * (1 - self.reconWeight) * jerkWeight + jointAcc * (1 - self.reconWeight) * (1 - jerkWeight) + oloss * self.reconWeight
        elif self.perpLossType == 'bone':

            adboneVecs = adData[:, :, self.neighbor_link[:, 0], :] - adData[:, :, self.neighbor_link[:, 1], :]
            adboneLengths = torch.sum(torch.mul(adboneVecs, adboneVecs), axis=-1)

            refboneVecs = refData[:, :, self.neighbor_link[:, 0], :] - refData[:, :, self.neighbor_link[:, 1], :]
            refboneLengths = torch.sum(torch.mul(refboneVecs, refboneVecs), axis=-1)

            diff = refboneLengths - adboneLengths
            boneLengthsLoss = torch.mean(torch.sum(torch.sum(torch.mul(diff, diff), axis=-1), axis=-1), axis=-1)

            return boneLengthsLoss * (1 - self.reconWeight) + oloss * self.reconWeight

        elif self.perpLossType == 'acc-bone':

            refAcc = (refData[:, 2:, :, :] - 2 * refData[:, 1:-1, :, :] + refData[:, :-2, :, :]) / self.deltaT / self.deltaT

            adAcc = (adData[:, 2:, :, :] - 2 * adData[:, 1:-1, :, :] + adData[:, :-2, :, :]) / self.deltaT / self.deltaT

            diff = refAcc - adAcc

            squaredLoss = torch.sum(torch.mul(diff, diff),axis=-1)

            weightedSquaredLoss = squaredLoss * self.jointWeights

            jointAcc = torch.mean(torch.sum(torch.sum(weightedSquaredLoss, axis=-1), axis=-1), axis=-1)

            adboneVecs = adData[:, :, self.neighbor_link[:,0], :] - adData[:, :, self.neighbor_link[:,1], :]
            adboneLengths = torch.sum(torch.mul(adboneVecs, adboneVecs), axis=-1)

            refboneVecs = refData[:, :, self.neighbor_link[:, 0], :] - refData[:, :, self.neighbor_link[:, 1], :]
            refboneLengths = torch.sum(torch.mul(refboneVecs, refboneVecs), axis=-1)

            diff = refboneLengths -adboneLengths
            boneLengthsLoss = torch.mean(torch.sum(torch.sum(torch.mul(diff, diff), axis=-1), axis=-1), axis=-1)

            return boneLengthsLoss * (1 - self.reconWeight) * self.boneLenWeight + jointAcc * (1 - self.reconWeight) * (1 - self.boneLenWeight) + oloss * self.reconWeight