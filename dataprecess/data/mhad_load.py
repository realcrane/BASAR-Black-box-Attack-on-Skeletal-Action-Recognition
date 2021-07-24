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

val_data_path = '/usr/not-backed-up/dyf/code/data/mhad/val_data.npy'
val_label_path = '/usr/not-backed-up/dyf/code/data/mhad/val_label.pkl'
val_data = np.load(val_data_path)
target_data = []
target_labels = []
target_names = []
with open(val_label_path, 'rb') as f:
    val_name, val_label = pickle.load(f)
val_unique,val_counts = np.unique(val_label,return_counts=True)

print('1')