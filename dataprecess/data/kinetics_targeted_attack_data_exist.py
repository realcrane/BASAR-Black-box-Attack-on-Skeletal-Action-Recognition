import os
from datetime import datetime
import pickle
import numpy as np
from gekko import GEKKO
import shutil
import sys
sys.path.append('../')
from feeder.feeder import Feeder
from net.stgcn.stgcn import Model
from copy import deepcopy
from tools.bone_constraints import bone_constraints
from tools.tools import forward_perturbation
from tools.tools import get_diff
from tools.tools import orthogonal_perturbation
import tools.tools as tools
import tools.kinematics as kinematics
import torch
import torch.nn as nn
import tools.spherical_system as sphere
sys.path.append('../motion')
from Quaternions import Quaternions
import BVH as BVH
import Animation as Animation
import tools.optimization as op
import tools.tools as tools
from tools.bone_constraints import bone_constraints_np
from motion.InverseKinematics import JacobianInverseKinematics

def main():
    torch.cuda.set_device(0)
    model = Model(in_channels=3, num_class=400, edge_importance_weighting=True,
                  graph_args={'layout': 'openpose', 'strategy': 'spatial'})
    phfile = '/usr/not-backed-up/dyf/code/black-box-attack/models/stgcn/st_gcn.kinetics.pt'
    '''
    train_data_path = '/usr/not-backed-up/dyf/code/data/kinetics/val_data.npy'
    train_label_path = '/usr/not-backed-up/dyf/code/data/kinetics/val_label.pkl'
    val_data_path = '/usr/not-backed-up/dyf/code/data/kinetics/val_data.npy'
    val_label_path = '/usr/not-backed-up/dyf/code/data/kinetics/val_label.pkl'
    '''
    train_data_path = '/usr/not-backed-up/dyf/code/MS-G3D/data/kinetics/train_data_joint.npy'
    train_label_path = '/usr/not-backed-up/dyf/code/MS-G3D/data/kinetics/val_label.pkl'
    val_data_path = '/usr/not-backed-up/dyf/code/MS-G3D/data/kinetics/val_data_joint.npy'
    val_label_path = '/usr/not-backed-up/dyf/code/MS-G3D/data/kinetics/val_label.pkl'
    model.load_state_dict(torch.load(phfile))
    model.eval()
    model.cuda()
    targets = np.load(val_data_path)
    with open(val_label_path, 'rb') as f:
        sample_names, labels = pickle.load(f)


    # data_size: nX3Xnum_framesX25X2
    target_samples = []
    target_classes = []
    initial_samples = []
    initial_classes = []
    adversarial_samples = []
    ad_ik_samples = []
    attack_classes_ik = []
    attack_classes = []
    n_calls_ik = []
    n_calls_list = []
    n_steps_list = []
    diffes = []
    diffes_post = []
    diffes_ik_post = []
    w1s = []
    n_target = 0

    for i in range(0,len(targets)):
        target = targets[i:i+1]
        label = labels[i]
        if np.any(target[:, :, :,:,1] != 0):
            continue
        target_sample_data= torch.from_numpy(target).float().cuda()
        target_class = torch.tensor(label).long().cuda()
        with torch.no_grad():
            target_output = model(target_sample_data).cuda()
            _, target_class_old = torch.max(target_output, 1)
        print('1')
    path_save = '/usr/not-backed-up/dyf/code/data/experiments/hdm05/tar_op/' + "target_val_3classes"
    os.makedirs(path_save)
    np.save(os.path.join(path_save, 'target_classes.npy'), np.array(target_classes))
    np.save(os.path.join(path_save, 'initial_classes.npy'), np.array(initial_classes))
    np.save(os.path.join(path_save, 'initial_samples.npy'), np.array(initial_samples))
    np.save(os.path.join(path_save, 'target_samples.npy'), np.array(target_samples))
    print('1')



if __name__ == '__main__':
    main()