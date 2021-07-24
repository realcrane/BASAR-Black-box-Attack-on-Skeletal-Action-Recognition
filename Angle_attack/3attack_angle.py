import os
import pickle
import numpy as np
from gekko import GEKKO
import shutil
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
import sys
sys.path.append('../motion')
from Quaternions import Quaternions
import BVH as BVH
import Animation as Animation
import tools.tools as tools
from tools.bone_constraints import bone_constraints_np
from motion.InverseKinematics import JacobianInverseKinematics
# read me: op only position, no IK
def matlab_transpose(data):
    data = np.transpose(data, [0, 2, 1, 3])
    matlab_data = data.reshape((-1, 60, 75), order='F')
    return matlab_data

def main():
    # model
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.set_device(1)
    '''
    #mhad
    phfile = '/usr/not-backed-up/dyf/code/black-box-attack/models/stgcn/mhad_epoch10_model.pt'
    val_data_path = '/usr/not-backed-up/dyf/code/data/mhad/val_data.npy'
    val_label_path = '/usr/not-backed-up/dyf/code/data/mhad/val_label.pkl'
    '''
    model = Model(in_channels=3, num_class=65, dropout=0.5, edge_importance_weighting=True,
                  graph_args={})


    phfile = '/usr/not-backed-up/dyf/code/black-box-attack/models/stgcn/hdm05_epoch65_model.pt'
    train_data_path = '/usr/not-backed-up/dyf/code/data/hdm05/train_data.npy'
    train_label_path = '/usr/not-backed-up/dyf/code/data/hdm05/train_label.pkl'
    val_data_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_data.npy'
    val_label_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_label.pkl'
    model.load_state_dict(torch.load(phfile))
    model.eval()
    model.cuda()
    torch.set_printoptions(precision=8)
    # data_size: nX3X60X25X2
    '''
    T = Feeder(train_data_path, train_label_path, random_choose=False, random_move=False,
               window_size=-1, mmap=True, debug=False)
    Initial_data = torch.utils.data.DataLoader(T, batch_size=1, shuffle=True, drop_last=False)
    '''
    T = Feeder(train_data_path, train_label_path, random_choose=True, random_move=False,
               window_size=-1, mmap=True, debug=False)
    V = Feeder(val_data_path, val_label_path, random_choose=False, random_move=False,
               window_size=-1, mmap=True, debug=False)
    Initial_data = torch.utils.data.DataLoader(V, batch_size=1, shuffle=True, drop_last=False)
    Target_data = torch.utils.data.DataLoader(V, batch_size=1, shuffle=False, drop_last=False)
    core = np.load('../dataprecess/core/hdm05_core_ik.npy')

    target_samples = []
    target_classes = []
    initial_samples = []
    initial_classes = []
    adversarial_samples = []
    attack_classes = []
    diffes = []
    diffes_euler = []
    n_target = 0
    n_calls_list = []
    n_steps_list = []
    for target_sample_data_old, target_label_old in Target_data:
        target_sample_data_old = target_sample_data_old.float().cuda()
        target_label_old = target_label_old.long().cuda()
        with torch.no_grad():
            target_output = model(target_sample_data_old).cuda()
            u, target_class_old = torch.max(target_output, 1)
        if target_class_old != target_label_old:
            continue
        n_target += 1
        if n_target > 1:
            break
        n_initial = 0
        target_sample_data = target_sample_data_old
        target_class = target_class_old
        target_sample = target_sample_data[0, :, :, :, 0]
        targets_sample_postprocessing = tools.postprocessing(target_sample.cpu().numpy(), core)
        target_sample_ik,target_sample_euler = kinematics.target_ik(targets_sample_postprocessing)
        target_samples.append(deepcopy(target_sample_data[0].cpu().numpy()))
        target_classes.append(deepcopy(target_class.cpu().numpy()))

        for initial_sample_data_old, initial_label_old in Initial_data:


            # get initial_data
            initial_sample_data_old = initial_sample_data_old.float().cuda()
            initial_label_old = initial_label_old.long().cuda()
            with torch.no_grad():
                initial_output = model(initial_sample_data_old).cuda()
            _, initial_class_old = torch.max(initial_output, 1)
            if initial_class_old != initial_label_old:
                continue
            if target_class == initial_class_old:
                continue
            n_initial += 1
            if n_initial > 1:
                break
            initial_sample_data = initial_sample_data_old
            initial_class = initial_class_old
            initial_sample_data = initial_sample_data
            initial_sample = initial_sample_data[0, :, :, :, 0]
            initial_sample_postprocessing = tools.postprocessing(initial_sample.cpu().numpy(), core)
            initial_sample_ik, initial_sample_euler = kinematics.target_ik(initial_sample_postprocessing)
            initial_classes.append(deepcopy(initial_class.cpu().numpy()))
            initial_samples.append(deepcopy(initial_sample.cpu().numpy()))
            # initial_sample = np.transpose(initial_sample, [0, 2, 1])  # (3,60,25) (3,25,60)
            n_steps = 0
            n_calls = 0
            epsilon = 1.
            delta = 0.1
            theta = 0.8
            adversarial_samples_out = []
            adversarial_samples_out_euler = []
            attack_classes_out = []
            diffes_out = []
            diffes_out_euler = []
            while True:
                #f_p = forward_perturbation(epsilon * get_diff(initial_sample, target_sample), initial_sample, target_sample)
                f_p = tools.forward_perturbation_angle(epsilon * tools.get_diff_angle(initial_sample_euler, target_sample_euler),
                                                       initial_sample_euler, target_sample_euler)
                trial_sample_euler = initial_sample_euler + f_p  # (1,3,60,25)
                trial_sample_euler[:,0:2,:] = target_sample_euler[:,0:2,:]
                trial_sample_ik = initial_sample_ik.copy()
                trial_sample_ik = kinematics.animation_euler(trial_sample_euler,trial_sample_ik)
                trial_sample_postprocess = Animation.positions_global(trial_sample_ik)
                trial_sample = torch.from_numpy(tools.normalize(trial_sample_postprocess[:,1:,:], core)).float().cuda()
                trial_sample_data = initial_sample_data.clone()
                trial_sample_data[0, :, :, :, 0] = trial_sample
                trial_output = model(trial_sample_data)
                _, trial_class = torch.max(trial_output, 1)
                n_calls += 1
                if trial_class != target_class:
                    adversarial_sample = trial_sample  # (1,3,60,25)
                    adversarial_sample_ik = trial_sample_ik; adversarial_sample_euler = trial_sample_euler
                    attack_class = trial_class
                    break
                else:
                    epsilon *= 0.9
            adversarial_sample = adversarial_sample.reshape(3, 60, 25)
            # adversarial_sample = np.transpose(adversarial_sample, [0, 2, 1])
            while True:
                print("Step #{}...".format(n_steps))
                print("\tDelta step...")
                d_step = 0
                n_op_break = 0
                while True:
                    d_step += 1
                    print("\t#{}".format(d_step))
                    trial_samples = []
                    trial_samples_euler = []
                    predictions = []
                    adversarial_samples_op = []
                    adversarial_samples_op_euler = []
                    delta_op = []
                    n_break = 0
                    if d_step >= 200:
                        adversarial_samples.append(deepcopy(adversarial_sample.cpu().numpy()))
                        attack_classes.append(deepcopy(attack_class.cpu().numpy()))
                        diff = torch.mean(get_diff(adversarial_sample.permute(1, 2, 0), target_sample.permute(1, 2, 0)))
                        diffes.append(deepcopy(diff.cpu().numpy()))
                        diffes_euler.append(deepcopy(diff_euler))
                        n_calls_list.append(deepcopy(n_calls))
                        n_steps_list.append(deepcopy(n_steps))
                        n_op_break = 1
                        break
                    for i in torch.arange(10):
                        '''
                        or_p = tools.orthogonal_perturbation_angle(delta, adversarial_sample_euler, target_sample_euler)
                        trial_sample_euler = adversarial_sample_euler + or_p
                        trial_sample_ik = kinematics.animation_euler(trial_sample_euler,trial_sample_ik)
                        trial_sample_postprocess = Animation.positions_global(trial_sample_ik)
                        trial_sample = torch.from_numpy(tools.normalize(trial_sample_postprocess[:,1:,], core)).float().cuda()
                        '''
                        or_p = orthogonal_perturbation(delta, adversarial_sample, target_sample)
                        trial_sample = adversarial_sample + or_p
                        trial_sample_data[0, :, :, :, 0] = trial_sample
                        trial_output = model(trial_sample_data)
                        _, prediction = torch.max(trial_output, 1)
                        predictions.append(deepcopy(prediction))
                        trial_samples.append(deepcopy(trial_sample))
                        trial_samples_euler.append(deepcopy(trial_sample_euler))
                    n_calls += 10
                    predictions = torch.tensor(predictions).cuda()
                    predictions_value = torch.unique(predictions)
                    prediction_len = len(predictions_value)
                    attack_len = prediction_len
                    for i in torch.arange(prediction_len):
                        if predictions_value[i] == target_class:
                            attack_len -= 1
                            m_target = (predictions == target_class).float()
                            t_score = torch.mean(m_target)
                            if t_score == 1:
                                delta *= 0.9
                                break
                        else:
                            n_break = 1
                            m = (predictions == predictions_value[i]).float()
                            d_score = torch.mean(m)
                            if d_score > 0.0:
                                if d_score < 0.3:
                                    delta_op.append(deepcopy(delta*0.9))
                                elif d_score > 0.7:
                                    delta_op.append(deepcopy(delta/0.9))
                                    if delta_op[-1] >= theta:
                                        delta_op[-1] = theta
                                else:
                                    delta_op.append(deepcopy(delta))
                                adversarial_sample = trial_samples[torch.where(predictions == predictions_value[i])[0][0]]
                                adversarial_sample_postprocess = tools.postprocessing(adversarial_sample.cpu().numpy(), core)
                                adversarial_sample_ik, adversarial_sample_euler = kinematics.target_ik(adversarial_sample_postprocess)
                                ad_sample2_postprocess = Animation.positions_global(adversarial_sample_ik)
                                ad_sample2_postprocess[:, [1, 6, 11], :] = adversarial_sample_postprocess[:,
                                                                                    [0, 5, 10], :]
                                ad_sample2 = tools.normalize(ad_sample2_postprocess[:,1:,:],core)
                                ad_sample = adversarial_sample.cpu().numpy()
                                ad_sample2 = torch.from_numpy(ad_sample2).float().cuda()
                                ad_sample2_data = torch.zeros_like(target_sample_data)
                                ad_sample2_data[0,:,:,:,0] = ad_sample2
                                ad_output = model(ad_sample2_data)
                                _,ad_class = torch.max(ad_output, 1)
                                adversarial_samples_op.append(deepcopy(adversarial_sample))
                                adversarial_samples_op_euler.append(deepcopy(adversarial_sample_euler))

                    if n_break == 1:
                        break
                if n_op_break == 1:
                    break

                print("\tEpsilon step...")
                e_step = 0
                attack_class_op = []
                diff_op = []
                diff_op_euler = []

                epsilon_op = epsilon*np.ones(attack_len)
                for i in torch.arange(attack_len):
                    while True:
                        e_step += 1
                        print("\t#{}".format(e_step))
                        adversarial_samples_op[i] = adversarial_samples_op[i].reshape(3, 60, 25)
                        trial_sample_euler = adversarial_samples_op_euler[i] + tools.forward_perturbation_angle(epsilon_op[i] * tools.get_diff_angle(adversarial_samples_op_euler[i], target_sample_euler),
                                                                                 adversarial_samples_op_euler[i], target_sample_euler)
                        trial_sample_ik = kinematics.animation_euler(trial_sample_euler,trial_sample_ik)
                        trial_sample_postprocess = Animation.positions_global(trial_sample_ik)
                        trial_sample = torch.from_numpy(tools.normalize(trial_sample_postprocess[:,1:,], core)).float().cuda()
                        #trial_sample = np.transpose(trial_sample, [0, 2, 1])
                        trial_sample[:,:,[0,5,10]] = adversarial_samples_op[i][:,:,[0,5,10]]
                        trial_sample_data[0, :, :, :, 0] = trial_sample
                        trial_output = model(trial_sample_data)
                        n_calls += 1
                        _, trial_class = torch.max(trial_output, 1)
                        if trial_class != target_class:
                            adversarial_samples_op[i] = trial_samples[0]
                            attack_class_op.append(deepcopy(trial_class))
                            print("Epsilon: {}".format(epsilon_op[i]))
                            if epsilon_op[i] < 1:
                                epsilon_op[i] /= 0.5
                                if epsilon_op[i] >= 1:
                                    epsilon_op[i] = 0.9
                            elif epsilon_op[i] >= 1:
                                epsilon_op[i] = 0.9
                            break
                        elif e_step > 500:
                            break
                        else:
                            epsilon_op[i] *= 0.5
                    di_euler =np.abs(adversarial_samples_op_euler - target_sample_euler)
                    diff_trial_euler = np.mean(np.abs(adversarial_samples_op_euler - target_sample_euler))
                    diff_op_euler.append(deepcopy(diff_trial_euler))
                    diff_trial = torch.mean(get_diff(adversarial_samples_op[i].permute(1,2,0), target_sample.permute(1,2,0)))
                    diff_op.append(deepcopy(diff_trial))
                index_op_np = np.array(diff_op_euler).argmin()
                index_op = torch.tensor(index_op_np)
                attack_class = attack_class_op[index_op]
                adversarial_sample = adversarial_samples_op[index_op]
                diff = diff_op[index_op]
                diff_euler = diff_op_euler[index_op_np]
                epsilon = epsilon_op[index_op]
                delta = delta_op[index_op]
                # argmin
                adversarial_samples_out.append(deepcopy(adversarial_sample))
                adversarial_samples_out_euler.append(deepcopy(adversarial_sample_euler))
                attack_classes_out.append(deepcopy(attack_class))
                diffes_out.append(deepcopy(diff))
                diffes_out_euler.append(deepcopy(diff_euler))


                n_steps += 1
                if diff_euler <= 1 or e_step > 500:
                    trial_sample_bone = adversarial_samples_op[index_op] + forward_perturbation(0.5*epsilon_op[index_op] * get_diff(adversarial_samples_op[index_op], target_sample),
                                                                                                    adversarial_samples_op[index_op], target_sample)
                    trial_sample_bone_data = trial_sample_data.clone()
                    print("{} steps".format(n_steps))
                    print("Mean Squared Error: {}".format(diff))
                    print("Mean Angles Error: {}".format(diff_euler))
                    print("Calls: {}".format(n_calls))
                    print("Attack Class: {}".format(attack_class))
                    print("Target Class: {}".format(target_class))
                    print("Delta: {}".format(delta))
                    print("initial class: {}".format(initial_class))
                    adversarial_samples.append(deepcopy(adversarial_sample.cpu().numpy()))
                    attack_classes.append(deepcopy(attack_class.cpu().numpy()))
                    diffes.append(deepcopy(diff.cpu().numpy()))
                    diffes_euler.append(deepcopy(diff_euler))

                    n_calls_list.append(deepcopy(n_calls))
                    n_steps_list.append(deepcopy(n_steps))
                    break
                if n_steps >= 20:
                    index_out = torch.tensor(diffes_out).argmin()
                    adversarial_sample = adversarial_samples_out[index_out]
                    attack_class = attack_classes_out[index_out]
                    diff = diffes_out[index_out]
                    print("{} steps".format(n_steps))
                    print("Mean Squared Error: {}".format(diff))
                    print("Mean Angles Error: {}".format(diff_euler))
                    print("Calls: {}".format(n_calls))
                    print("Attack Class: {}".format(attack_class))
                    print("Target Class: {}".format(target_class))
                    print("Delta: {}".format(delta))
                    adversarial_samples.append(deepcopy(adversarial_sample.cpu().numpy()))
                    attack_classes.append(deepcopy(attack_class.cpu().numpy()))
                    diffes.append(deepcopy(diff.cpu().numpy()))
                    diffes_euler.append(deepcopy(diff_euler))
                    n_calls_list.append(deepcopy(n_calls))
                    n_steps_list.append(deepcopy(n_steps))
                    break

                print("Mean Squared Error: {}".format(diff))
                print("Mean Angles Error: {}".format(diff_euler))
                print("Calls: {}".format(n_calls))
                print("Attack Class: {}".format(attack_class))
                print("Target Class: {}".format(target_class))
                print("Initial Class:{}".format(initial_class))
                print("Delta: {}".format(delta))
                print('n target:: {}'.format(n_target))
    diff_test1 = torch.mean(get_diff(adversarial_sample.permute(1, 2, 0), target_sample.permute(1, 2, 0)))
    adversarial_samples = np.array(adversarial_samples)
    initial_samples =np.array(initial_samples)
    target_samples = np.array(target_samples)
    attack_classes = np.array(attack_classes)
    target_classes = np.array(target_classes)
    initial_classes = np.array(initial_classes)

    path_save = '/usr/not-backed-up/dyf/code/black-box-attack/data/trial_hdm05/1e-1/'
    np.savetxt(os.path.join(path_save,'target_classes.txt'), target_classes)
    np.savetxt(os.path.join(path_save, 'initial_classes.txt'), initial_classes)
    np.savetxt(os.path.join(path_save, 'attack_classes1e-1.txt'), attack_classes)
    np.save(os.path.join(path_save,'adversarial_samples1e-1.npy'),adversarial_samples)
    np.save(os.path.join(path_save,'initial_samples.npy'), initial_samples)
    np.save(os.path.join(path_save,'target_samples.npy'), target_samples)
    adversarial_samples_matlab = matlab_transpose(adversarial_samples)
    target_samples_matlab = matlab_transpose(target_samples[:,:,:,:,0])
    target_samples_matlab, adversarial_samples_matlab = tools.postmatlab(target_samples_matlab,adversarial_samples_matlab,'hdm05')
    np.save(os.path.join(path_save,'target_samples_matlab.npy'), target_samples_matlab)
    np.save(os.path.join(path_save,'adversarial_samples1e-1_matlab.npy'), adversarial_samples_matlab)
    print('1')

if __name__ == '__main__':
    main()