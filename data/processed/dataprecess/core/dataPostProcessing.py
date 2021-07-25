import numpy as np
import pdb
import os
import shutil
import glob
hdm05 = np.load('./hdm05_core_ik.npy')
mhad = np.load('./mhad_core_ik.npy')
dataset = 'mhad'
if dataset == 'hdm05':
    core = np.load('./hdm05_core.npy')
    core_ik = core.reshape((-1, 3, 25), order='F')
    core_ik = np.transpose(core_ik,[0,2,1])
    np.save('./hdm05_core_ik.npy',core_ik)
if dataset == 'mhad':
    core = np.load('./mhad_core.npy')
    core_ik = core.reshape((-1, 3, 25), order='F')
    core_ik = np.transpose(core_ik,[0,2,1])
    np.save('./mhad_core_ik.npy',core_ik)
print('1')