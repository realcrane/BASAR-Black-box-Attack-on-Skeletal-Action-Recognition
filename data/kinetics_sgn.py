import numpy as np
import tools.tools as tools
import pickle
import os
import gc
train_data_path = '/usr/not-backed-up/dyf/code/MS-G3D/data/kinetics/train_data_joint.npy'
train_label_path = '/usr/not-backed-up/dyf/code/MS-G3D/data/kinetics/train_label.pkl'
val_data_path = '/usr/not-backed-up/dyf/code/MS-G3D/data/kinetics/val_data_joint.npy'
val_label_path = '/usr/not-backed-up/dyf/code/MS-G3D/data/kinetics/val_label.pkl'
path_save = '/usr/not-backed-up/dyf/code/data/kinetics/sgn/'
with open(val_label_path, 'rb') as f:
    val_sample_name, val_label = pickle.load(f)
with open(train_label_path, 'rb') as e:
    train_sample_name, train_label = pickle.load(e)

val_label = np.array(val_label).astype("int64")
train_label= np.array(train_label).astype("int64")
train_data = np.load(train_data_path).astype("float32")#[:10]
val_data = np.load(val_data_path).astype("float32")#[:10]
train_num = len(train_data)
val_num = len(val_data)
'''
train_data = np.zeros((train_num,3,300,25,2)).astype("float32")
val_data = np.zeros((val_num,3,300,25,2)).astype("float32")
train_data[:,:,:,:18,:]= train_data_old
val_data[:,:,:,:18,:] = val_data_old
del train_data_old
del val_data_old
'''
gc.collect()
train_sgn1 = tools.matlab_transpose(train_data[:,:,:,:,0],joints=54).astype("float32")
train_sgn2 = tools.matlab_transpose(train_data[:,:,:,:,1],joints=54).astype("float32")
val_sgn1 = tools.matlab_transpose(val_data[:,:,:,:,0],joints=54).astype("float32")
val_sgn2 = tools.matlab_transpose(val_data[:,:,:,:,1],joints=54).astype("float32")
del train_data
del val_data
gc.collect()
train_zero = np.zeros((train_num,300,108)).astype("float32")
val_zero = np.zeros((val_num,300,108)).astype("float32")
train_zero[:,:,:54] = train_sgn1
train_zero[:,:,54:] = train_sgn2
val_zero[:,:,:54] = val_sgn1
val_zero[:,:,54:] = val_sgn2
np.save(os.path.join(path_save, 'train_label.npy'), np.array(train_label))
np.save(os.path.join(path_save, 'val_label.npy'), np.array(val_label))
np.save(os.path.join(path_save, 'val_sgn_data2.npy'), np.array(val_zero))
np.save(os.path.join(path_save, 'train_sgn_data2.npy'), np.array(train_zero))
print('1')
