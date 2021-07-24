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

train_data_path = '/usr/not-backed-up/dyf/code/data/hdm05/train_data.npy'
train_label_path = '/usr/not-backed-up/dyf/code/data/hdm05/train_label.pkl'
val_data_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_data.npy'
val_label_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_label.pkl'
train_data = np.load(train_data_path)
with open(train_label_path, 'rb') as f:
    train_name, train_label = pickle.load(f)
val_data = np.load(val_data_path)
with open(val_label_path, 'rb') as f:
    val_name, val_label = pickle.load(f)
train_unique,train_counts = np.unique(train_label,return_counts=True)
val_unique,val_counts = np.unique(val_label,return_counts=True)
print('1')