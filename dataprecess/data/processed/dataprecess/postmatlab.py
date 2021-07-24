import numpy as np
import pdb
import os
import shutil
import glob

def postmatlab(orMotions,motions,datatype):
    if datatype == 'hdm05':
        core = np.load('./core/hdm05_core.npy')
    elif datatype == 'mhad':
        core = np.load('./core/mhad_core.npy')
    elif datatype == 'ntu':
        core = np.load('./core/ntu_core.npy')
    std = core[0]
    mean = core[1]
    for i in range(len(motions)):
        motions[i] = motions[i] * std + mean
        orMotions[i] = orMotions[i] * std + mean
    return orMotions, motions


def postmatlab_one(motions,datatype):
    if datatype == 'hdm05':
        core = np.load('./core/hdm05_core.npy')
    elif datatype == 'mhad':
        core = np.load('./core/mhad_core.npy')
    elif datatype == 'ntu':
        core = np.load('./core/ntu_core.npy')
    std = core[0]
    mean = core[1]
    for i in range(len(motions)):
        motions[i] = motions[i] * std + mean
    return  motions





