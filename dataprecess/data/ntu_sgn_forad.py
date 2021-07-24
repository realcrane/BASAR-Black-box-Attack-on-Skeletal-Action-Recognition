import numpy as np
import tools.tools as tools
import pickle
import os
aim = 'npy'
if aim == 'pkl':
    path_load = '/home/dyf/code/datasets/hdm05/clean_gaussian/sigma0.5/'
    train_data_path = path_load + 'train_data.npy'
    train_label_path = path_load + 'train_label.pkl'
    path_save = '/home/dyf/code/datasets/hdm05/sgn/clean_gaussian/sigma0.5/'
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    with open(train_label_path, 'rb') as e:
        train_sample_name, train_label = pickle.load(e)
    train_label = np.array(train_label).astype("int64")
    train_data = np.load(train_data_path)[:,:,:,:,0].astype("float32")
else:
    path_load = '/home/dyf/code/datasets/ntu/sgn/adv_training/nompiter1/'
    train_data_path = path_load + 'adversarial_ik_samples.npy'
    train_label_path = path_load + 'target_classes.npy'
    path_save = '/home/dyf/code/datasets/ntu/sgn/adv_training/nompiter1/'
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    train_label = np.load(train_label_path).reshape((-1)).astype(np.int64)
    np.save(os.path.join(path_save, 'train_label.npy'), np.array(train_label))
    train_data = np.load(train_data_path)[:,:,:,:].astype(np.float32)
train_sgn = tools.matlab_transpose(train_data[:,:,:,:,0])
train_sgn2 = tools.matlab_transpose(train_data[:,:,:,:,1])
train_zero = np.zeros((len(train_data),300,150)).astype(np.float32)
train_zero[:,:,:75] = train_sgn
train_zero[:,:,75:] = train_sgn2

np.save(os.path.join(path_save, 'train_sgn_data.npy'), np.array(train_zero))
print('1')
