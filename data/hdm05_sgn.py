import numpy as np
import tools.tools as tools
import pickle
import os
path_load = '/home/dyf/code/datasets/hdm05/'
train_data_path = path_load + 'train_data.npy'
train_label_path = path_load + 'train_label.pkl'
val_data_path = path_load +'val_data.npy'
val_label_path = path_load +'val_label.pkl'
path_save = '/home/dyf/code/datasets/hdm05/sgn/'
with open(val_label_path, 'rb') as f:
    val_sample_name, val_label = pickle.load(f)
with open(train_label_path, 'rb') as e:
    train_sample_name, train_label = pickle.load(e)
val_label = np.array(val_label).astype("int64")
train_label = np.array(train_label).astype("int64")
train_data = np.load(train_data_path)[:,:,:,:,0].astype("float32")
val_data = np.load(val_data_path)[:,:,:,:,0].astype("float32")
train_sgn = tools.matlab_transpose(train_data)
val_sgn = tools.matlab_transpose(val_data)
train_zero = np.zeros((2122,60,150)).astype("float32")
val_zero = np.zeros((531,60,150)).astype("float32")
train_zero[:,:,:75] = train_sgn
val_zero[:,:,:75] = val_sgn
np.save(os.path.join(path_save, 'train_label.npy'), np.array(train_label))
np.save(os.path.join(path_save, 'val_label.npy'), np.array(val_label))
np.save(os.path.join(path_save, 'val_sgn_data.npy'), np.array(val_zero))
np.save(os.path.join(path_save, 'train_sgn_data.npy'), np.array(train_zero))
print('1')
