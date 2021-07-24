import numpy as np
import tools.tools as tools
import pickle
import os
aim = 'pkl'
if aim == 'pkl':
    path_load = '/home/dyf/code/backups_cvpr2021_attack_motion/advtraining/gaussian/hdm05/sigma0.1/'
    train_data_path = path_load + 'gaussian_samples.npy'
    train_label_path = path_load + 'gaussian_labels.pkl'
    path_save = '/home/dyf/code/backups_cvpr2021_attack_motion/advtraining/gaussian/hdm05/sigma0.1/'
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    with open(train_label_path, 'rb') as e:
        train_sample_name, train_label = pickle.load(e)
    train_label = np.array(train_label).astype("int64")
    train_data = np.load(train_data_path)[:,:,:,:,0].astype("float32")
else:
    path_load = '/home/dyf/code/backups_cvpr2021_attack_motion/advtraining/gaussian/hdm05/sigma0.1/'
    train_data_path = path_load + 'gaussian_samples.npy'
    train_label_path = path_load + 'gaussian_labels.pkl'
    path_save = '/home/dyf/code/backups_cvpr2021_attack_motion/advtraining/gaussian/hdm05/sigma0.1/'
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    train_label = np.load(train_label_path).reshape((-1)).astype(np.int64)
    train_data = np.load(train_data_path)[:,:,:,:].astype("float32")
train_sgn = tools.matlab_transpose(train_data)

train_zero = np.zeros((len(train_data),60,150)).astype("float32")
train_zero[:,:,:75] = train_sgn

np.save(os.path.join(path_save, 'train_label.npy'), np.array(train_label))
np.save(os.path.join(path_save, 'train_sgn_data.npy'), np.array(train_zero))
print('1')
