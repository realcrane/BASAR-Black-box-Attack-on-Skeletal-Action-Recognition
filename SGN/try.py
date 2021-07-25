# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
import os.path as osp
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import h5py
import random
import os.path as osp
import sys
from six.moves import xrange
import math
import scipy.misc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
sys.path.append('../')
from SGN.model import SGN
from SGN.data import NTUDataLoaders, AverageMeter
import SGN.fit as fit
from SGN.util import make_dir, get_num_classes
import tools.tools as tools
w = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'
torch.cuda.set_device(3)
parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(
    network='SGN',
    dataset = 'NTU',
    case = 0,
    batch_size=1,
    max_epochs=120,
    monitor='val_acc',
    lr=0.001,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=1,
    print_freq = 20,
    train = 0,
    seg = 20,
    )
args = parser.parse_args()
ntu_loaders = NTUDataLoaders('NTU', 0, seg=20)
a = ntu_loaders.test_X[0,:,:75]
a_normal = tools.postprocessing_from_matlab(a)
a_mirror = tools.matlab_transpose(a_normal)
Target_data = torch.utils.data.DataLoader(ntu_loaders.test_set, batch_size=1, collate_fn=ntu_loaders.collate_fn_fix_test, shuffle=False, drop_last=False)
#Initial_data = torch.utils.data.DataLoader(ntu_loaders.test_set, batch_size=1, shuffle=True, drop_last=False)
args.num_classes = get_num_classes(args.dataset)
model = SGN(args.num_classes, args.dataset, args.seg, args)
phfile = '/usr/not-backed-up/dyf/code/SGN-master/results/NTU/SGN/0_best.pth'
#model.load_state_dict(torch.load(phfile))['state_dict']
save_path = '/usr/not-backed-up/dyf/code/SGN-master/results/NTU/SGN/'
checkpoint = osp.join(save_path, '%s_best.pth' % args.case)
model.load_state_dict(torch.load(checkpoint)['state_dict'])
model.eval()
model.cuda()


for target_sample_data_old, target_label_old in Target_data:
    target_sample_data_old = target_sample_data_old.float().cuda()
    target_label_old = target_label_old.long().cuda()
    with torch.no_grad():
        target_output = model(target_sample_data_old)
        #target_output = model(inputs.cuda())
        #target_output = target_output.view((-1, target_sample_data_old.size(0) // target_label_old.size(0), target_output.size(1)))
        #target_output = target_output.mean(1)
        _, target_class = torch.max(target_output, 1)
print('1')