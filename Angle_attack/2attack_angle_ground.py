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

# read me: op position final IK
def main():
    # model
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.set_device(1)
    dataset = 'hdm05'
    if dataset == 'ntu':
        phfile = '/usr/not-backed-up/dyf/code/black-box-attack/models/stgcn/st_gcn.ntu-xsub.pt'
        model = Model(in_channels=3, num_class=60, dropout=0.5, edge_importance_weighting=True,
                      graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'})
        train_data_path = '/usr/not-backed-up/dyf/code/data/ntu/xsub/val_data.npy'
        train_label_path = '/usr/not-backed-up/dyf/code/data/ntu/xsub/val_label.pkl'
        val_data_path = '/usr/not-backed-up/dyf/code/data/ntu/xsub/val_data.npy'
        val_label_path = '/usr/not-backed-up/dyf/code/data/ntu/xsub/val_label.pkl'
    if dataset == 'hdm05':
        model = Model(in_channels=3, num_class=65, dropout=0.5, edge_importance_weighting=True,
                      graph_args={})

        phfile = '/usr/not-backed-up/dyf/code/black-box-attack/models/stgcn/hdm05_epoch65_model.pt'
        train_data_path = '/usr/not-backed-up/dyf/code/data/hdm05/train_data.npy'
        train_label_path = '/usr/not-backed-up/dyf/code/data/hdm05/train_label.pkl'
        val_data_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_data.npy'
        val_label_path = '/usr/not-backed-up/dyf/code/data/hdm05/val_label.pkl'
    '''
    #mhad
    phfile = '/usr/not-backed-up/dyf/code/black-box-attack/models/stgcn/mhad_epoch10_model.pt'
    val_data_path = '/usr/not-backed-up/dyf/code/data/mhad/val_data.npy'
    val_label_path = '/usr/not-backed-up/dyf/code/data/mhad/val_label.pkl'
    '''
    model.load_state_dict(torch.load(phfile))
    model.eval()
    model.cuda()
    torch.set_printoptions(precision=8)
    # data_size: nX3Xnum_framesX25X2
    '''
    T = Feeder(train_data_path, train_label_path, random_choose=False, random_move=False,
               window_size=-1, mmap=True, debug=False)
    Initial_data = torch.utils.data.DataLoader(T, batch_size=1, shuffle=True, drop_last=False)
    '''

    T = Feeder(train_data_path, train_label_path, random_choose=False, random_move=False,
               window_size=-1, mmap=True, debug=False)
    V = Feeder(val_data_path, val_label_path, random_choose=False, random_move=False,
               window_size=-1, mmap=True, debug=False)
    Initial_data = torch.utils.data.DataLoader(T, batch_size=1, shuffle=True, drop_last=False)
    Target_data = torch.utils.data.DataLoader(V, batch_size=1, shuffle=False, drop_last=False)
    core = np.load('../dataprecess/core/hdm05_core_ik.npy')
    target_samples = []
    target_classes = []
    initial_samples = []
    initial_classes = []
    adversarial_samples = []
    adversarial_samples_ik = []
    attack_classes = []
    attack_classes_ik = []
    diffes = []
    diffes_post = []
    diffes_ik_post = []
    n_target = 0
    n_calls_list = []
    n_steps_list = []
    for target_sample_data_old, target_label_old in Target_data:
        target_sample_data_old = target_sample_data_old.float().cuda()
        target_label_old = target_label_old.long().cuda()
        with torch.no_grad():
            target_output = model(target_sample_data_old).cuda()
            _, target_class_old = torch.max(target_output, 1)
        if target_class_old != target_label_old:
            continue
        n_target += 1
        if n_target > 1:
            break
        n_initial = 0
        target_sample_data = target_sample_data_old
        if dataset == 'ntu':
            for p in range(0,299):
                if torch.all(target_sample_data[:,:,p,:,:] == 0):
                    break
            #target_sample_data = target_sample_data[:,:,:p,:,:]
        target_class = target_class_old
        target_sample = target_sample_data[0, :, :, :, 0]
        targets_sample_postprocessing = tools.postprocessing(target_sample.cpu().numpy(), core)
        target_sample_ik,target_sample_euler = kinematics.target_ik(targets_sample_postprocessing,True)
        num_frames = len(target_sample[0,:,0])
        #delete
        tar_sample_process2 = Animation.positions_global(target_sample_ik)
        tar_sample2 = torch.from_numpy(tools.normalize(tar_sample_process2[:,1:,:],core))
        target_sample_data[0,:,:,:,0] = tar_sample2
        #delete
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
            '''
            if dataset == 'ntu':
                initial_sample_data = initial_sample_data[:,:,:p,:,:]
            '''
            initial_sample = initial_sample_data[0, :, :, :, 0]
            initial_classes.append(deepcopy(initial_class.cpu().numpy()))
            initial_samples.append(deepcopy(initial_sample.cpu().numpy()))
            # initial_sample = np.transpose(initial_sample, [0, 2, 1])  # (3,num_frames,25) (3,25,num_frames)
            n_steps = 0
            n_calls = 0
            epsilon = 0.95
            delta = 0.1
            theta = 0.8
            ad = False
            adversarial_samples_out = []
            attack_classes_out = []
            diffes_out = []
            diffes_out_post = []

            while True:
                if dataset == 'ntu':
                    f_p = torch.zeros_like(target_sample).float().cuda()
                    f_p1 = forward_perturbation(epsilon * get_diff(initial_sample[:,:p,:], target_sample[:,:p,:]), initial_sample[:,:p,:], target_sample[:,:p,:])
                    f_p[:,:p,:] = f_p1
                else:
                    f_p = forward_perturbation(epsilon * get_diff(initial_sample, target_sample),
                                               initial_sample, target_sample)
                trial_sample = initial_sample + f_p  # (1,3,num_frames,25)
                trial_sample = trial_sample.reshape(1, 3, num_frames, 25)
                trial_sample_data = initial_sample_data.clone()
                trial_sample_data[:, :, :, :, 0] = trial_sample
                trial_output = model(trial_sample_data)
                _, trial_class = torch.max(trial_output, 1)
                n_calls += 1
                if trial_class != target_class:
                    adversarial_sample = trial_sample  # (1,3,num_frames,25)
                    attack_class = trial_class
                    break
                else:
                    epsilon *= 0.9

            adversarial_sample = adversarial_sample.reshape(3, num_frames, 25)
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
                    predictions = []
                    adversarial_samples_op = []
                    delta_op = []
                    n_break = 0
                    if d_step >= 200:
                        adversarial_samples.append(deepcopy(adversarial_sample.cpu().numpy()))
                        attack_classes.append(deepcopy(attack_class.cpu().numpy()))
                        diff = torch.mean(get_diff(adversarial_sample.permute(1, 2, 0), target_sample.permute(1, 2, 0)))
                        diffes.append(deepcopy(diff.cpu().numpy()))
                        n_calls_list.append(deepcopy(n_calls))
                        n_steps_list.append(deepcopy(n_steps))
                        n_op_break = 1
                        break
                    for i in torch.arange(10):
                        if dataset == 'ntu':
                            or_p = torch.zeros_like(target_sample).float().cuda()
                            or_p1 = orthogonal_perturbation(delta, adversarial_sample[:,:p,:], target_sample[:,:p,:],dataset)
                            or_p4 = or_p1.cpu().numpy()
                            or_p[:,:p,:] = or_p1
                            or_p5 = or_p.cpu().numpy()
                            #or_p = or_p.cpu().numpy()
                        else:
                            or_p = orthogonal_perturbation(delta, adversarial_sample, target_sample,dataset)
                        # trial_sample = np.transpose(adversarial_sample + or_p, [0, 2, 1])
                        #trial_sample = torch.from_numpy(np.transpose(adversarial_sample + or_p, [0, 2, 1])).reshape(1, 3, num_frames,25)
                        trial_sample = (adversarial_sample+or_p)#.reshape(1,3,num_frames,25)
                        trial_sample_data[0, :, :, :, 0] = trial_sample
                        trial_output = model(trial_sample_data)
                        _, prediction = torch.max(trial_output, 1)
                        predictions.append(deepcopy(prediction))
                        trial_samples.append(deepcopy(trial_sample))
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
                                adversarial_samples_op.append(deepcopy(adversarial_sample))
                    if n_break == 1:
                        break
                if n_op_break == 1:
                    break

                print("\tEpsilon step...")
                e_step = 0
                attack_class_op = []
                diff_op = []
                diff_op_post = []

                epsilon_op = epsilon*torch.ones(attack_len)
                for i in torch.arange(attack_len):
                    while True:
                        e_step += 1
                        print("\t#{}".format(e_step))
                        adversarial_samples_op[i] = adversarial_samples_op[i].reshape(3, num_frames, 25)
                        if dataset == 'ntu':
                            f_p2 = torch.zeros_like(target_sample).float().cuda()
                            f_p21 = forward_perturbation(epsilon_op[i] * get_diff(adversarial_samples_op[i][:,:p,:], target_sample[:,:p,:]),
                                adversarial_samples_op[i][:,:p,:], target_sample[:,:p,:])
                            f_p2[:,:p,:] = f_p21
                            #f_p2 = f_p2.cpu().numpy()
                            trial_sample = adversarial_samples_op[i] + f_p2
                        else:
                           trial_sample = adversarial_samples_op[i] + forward_perturbation(epsilon_op[i] * get_diff(adversarial_samples_op[i], target_sample),
                                                                                 adversarial_samples_op[i], target_sample)
                        #trial_sample = np.transpose(trial_sample, [0, 2, 1])
                        trial_sample = trial_sample.reshape(1, 3, num_frames, 25)
                        trial_sample_data[:, :, :, :, 0] = trial_sample
                        trial_output = model(trial_sample_data)
                        n_calls += 1
                        _, trial_class = torch.max(trial_output, 1)
                        if trial_class != target_class:
                            adversarial_samples_op[i] = trial_sample[0]
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
                    diff_trial = torch.mean(get_diff(adversarial_samples_op[i].permute(1,2,0), target_sample.permute(1,2,0)))
                    if dataset == 'hdm05':
                        ad_postprocess = tools.postprocessing(adversarial_sample.cpu().numpy(), core)
                        tar_postprocess = tools.postprocessing(target_sample.cpu().numpy(), core)
                        diff_trial_post = np.mean(tools.get_diff_np(ad_postprocess,tar_postprocess))
                        diff_trial_post = torch.tensor(diff_trial_post).cuda()
                    diff_op.append(deepcopy(diff_trial))
                    diff_op_post.append(deepcopy(diff_trial_post))
                index_op = torch.tensor(diff_op_post).argmin()
                attack_class = attack_class_op[index_op]
                adversarial_sample = adversarial_samples_op[index_op]
                diff = diff_op[index_op]
                diff_post = diff_op_post[index_op]
                epsilon = epsilon_op[index_op]
                delta = delta_op[index_op]
                # argmin
                adversarial_samples_out.append(deepcopy(adversarial_sample))
                attack_classes_out.append(deepcopy(attack_class))
                diffes_out.append(deepcopy(diff))
                diffes_out_post.append(deepcopy(diff_post))
                n_steps += 1
                print("Mean Squared Error: {}".format(diff))
                print("Mean Squared Post Error: {}".format(diff_post))
                print("Calls: {}".format(n_calls))
                print("Attack Class: {}".format(attack_class))
                print("Target Class: {}".format(target_class))
                print("Initial Class:{}".format(initial_class))
                print("Delta: {}".format(delta))
                print('n target:: {}'.format(n_target))
                if diff_post <= 2e-1 or e_step > 500:
                    ad = True
                    '''''
                    if ad_class == target_class:
                        for h in range(len(trial_samples)):
                            tr_sample = trial_samples[i]
                            tr_sample_postprocess = tools.postprocessing(tr_sample.cpu().numpy(), core)
                            tr_sample_ik, tr_sample_euler = kinematics.target_ik(tr_sample_postprocess)
                            tr_sample_postprocess2 = Animation.positions_global(tr_sample_ik)
                            tr_sample2 = torch.from_numpy(tools.normalize(tr_sample_postprocess2[:,1:,:],core)).cuda()
                            trial_sample_data[0,:,:,:,0] = tr_sample2
                            tr_output = model(trial_sample_data)
                            _, tr_class = torch.max(tr_output, 1)
                    '''''
                elif n_steps >= 5000:
                    ad = True
                    index_out = torch.tensor(diffes_out_post).argmin()
                    adversarial_sample = adversarial_samples_out[index_out]
                    attack_class = attack_classes_out[index_out]
                    diff = diffes_out[index_out]
                    diff_post = diffes_out_post[index_out]


                if ad == True:
                    #ik
                    adversarial_sample_postprocess = tools.postprocessing(adversarial_sample.cpu().numpy(), core)
                    adversarial_sample_ik, adversarial_sample_euler = kinematics.ad_ik(target_sample_ik,
                        adversarial_sample_postprocess)
                    adversarial_sample_postprocess2 = Animation.positions_global(adversarial_sample_ik)
                    ad_ik_original = torch.from_numpy(tools.normalize(adversarial_sample_postprocess2[:,1:,:], core)).float().cuda()
                    ad_ik_original_data = initial_sample_data.clone()
                    ad_ik_original_data[0,:,:,:,0] = ad_ik_original
                    epsilon = 1.0
                    #ik fp+orp
                    while True:
                        diff_trial_euler = np.mean(np.abs((adversarial_sample_euler - target_sample_euler)))
                        f_p = tools.forward_perturbation_angle(
                            epsilon * tools.get_diff_angle(adversarial_sample_euler, target_sample_euler),
                            adversarial_sample_euler, target_sample_euler)
                        trial_sample_euler = adversarial_sample_euler + f_p  # (1,3,60,25)
                        trial_sample_ik = kinematics.animation_euler(trial_sample_euler, adversarial_sample_ik)
                        trial_sample_process = Animation.positions_global(trial_sample_ik)
                        trial_sample = torch.from_numpy(
                            tools.normalize(trial_sample_process[:, 1:, :], core)).float().cuda()
                        trial_sample_data = initial_sample_data.clone()
                        trial_sample_data[0, :, :, :, 0] = trial_sample
                        trial_output = model(trial_sample_data)
                        _, trial_class = torch.max(trial_output, 1)
                        n_calls += 1
                        if trial_class != target_class:
                            ad_sample2 = trial_sample  # (1,3,60,25)
                            ad_sample2_process = tools.postprocessing(ad_sample2.cpu().numpy(),core,transpose=True)
                            adversarial_sample_ik2 = trial_sample_ik
                            adversarial_sample_euler2 = trial_sample_euler
                            ad_class_ik = trial_class
                            ik_ad_sample_process = trial_sample_process[:, 1:, :]
                            break
                        else:
                            epsilon *= 0.9
                        if epsilon < 0.01:
                            ad_sample2 = trial_sample  # (1,3,60,25)
                            adversarial_sample_ik = trial_sample_ik
                            adversarial_sample_euler2 = trial_sample_euler
                            ad_class_ik = trial_class
                            ik_ad_sample_process = trial_sample_process[:, 1:, :]
                            break
                    diff_trial_euler2 = np.mean(np.abs(adversarial_sample_euler2 - target_sample_euler))
                    ad_sample2 = ad_sample2.reshape(3, 60, 25)
                    diff_ik = np.mean(tools.get_diff_np(ik_ad_sample_process,
                                                        tools.postprocessing(target_sample.cpu().numpy(), core)))
                    '''''
                    ad_sample2_data = torch.zeros_like(target_sample_data)
                    ad_sample2_data[0, :, :, :, 0] = ad_sample2
                    ad_output = model(ad_sample2_data)
                    _, ad_class_ik = torch.max(ad_output, 1)
                    '''''
                    adversarial_samples.append(deepcopy(adversarial_sample.cpu().numpy()))
                    adversarial_samples_ik.append(deepcopy(ad_sample2.cpu().numpy()))
                    attack_classes.append(deepcopy(attack_class.cpu().numpy()))
                    attack_classes_ik.append(deepcopy(ad_class_ik.cpu().numpy()))
                    diffes.append(deepcopy(diff.cpu().numpy()))
                    diffes_post.append(deepcopy(diff_post.cpu().numpy()))
                    diffes_ik_post.append(deepcopy(diff_ik))
                    n_calls_list.append(deepcopy(n_calls))
                    n_steps_list.append(deepcopy(n_steps))
                    break




    diff_test1 = torch.mean(get_diff(adversarial_sample.permute(1, 2, 0), target_sample.permute(1, 2, 0)))
    adversarial_samples = np.array(adversarial_samples)
    adversarial_samples_ik = np.array(adversarial_samples_ik)
    initial_samples =np.array(initial_samples)
    target_samples = np.array(target_samples)
    attack_classes = np.array(attack_classes)
    attack_classes_ik = np.array(attack_classes_ik)
    target_classes = np.array(target_classes)
    initial_classes = np.array(initial_classes)
    ad_tar = bone_constraints_np(adversarial_samples, target_samples[:, :, :, :, 0])
    l_diff_op_ad, adLengths, tarLengths = ad_tar.bone_length()
    path_save = '/usr/not-backed-up/dyf/code/black-box-attack/data/trial_hdm05/2e-1/'
    np.savetxt(os.path.join(path_save,'target_classes.txt'), target_classes)
    np.savetxt(os.path.join(path_save, 'initial_classes.txt'), initial_classes)
    np.savetxt(os.path.join(path_save, 'attack_classes.txt'), attack_classes)
    np.savetxt(os.path.join(path_save, 'attack_classes_ik.txt'), attack_classes_ik)
    np.save(os.path.join(path_save,'adversarial_samples.npy'),adversarial_samples)
    np.save(os.path.join(path_save, 'adversarial_samples_ik.npy'), adversarial_samples_ik)
    np.save(os.path.join(path_save,'initial_samples.npy'), initial_samples)
    np.save(os.path.join(path_save,'target_samples.npy'), target_samples)
    adversarial_samples_matlab = tools.matlab_transpose(adversarial_samples)
    adversarial_samples_matlab_ik = tools.matlab_transpose(adversarial_samples_ik)
    target_samples_matlab = tools.matlab_transpose(target_samples[:,:,:,:,0])
    target_samples_matlab, adversarial_samples_matlab = tools.postmatlab(target_samples_matlab,adversarial_samples_matlab,dataset)
    adversarial_samples_matlab_ik = tools.postmatlab_one(adversarial_samples_matlab_ik, dataset)
    np.save(os.path.join(path_save,'target_samples_matlab.npy'), target_samples_matlab)
    np.save(os.path.join(path_save,'adversarial_samples_matlab.npy'), adversarial_samples_matlab)
    np.save(os.path.join(path_save, 'adversarial_samples_matlab_ik.npy'), adversarial_samples_matlab_ik)
    import scipy.io as io
    io.savemat(os.path.join(path_save,'target_samples_matlab.mat'), {'data': target_samples_matlab})
    io.savemat(os.path.join(path_save, 'adversarial_samples_matlab.mat'), {'data': adversarial_samples_matlab})
    io.savemat(os.path.join(path_save, 'adversarial_samples_matlab_ik.mat'), {'data': adversarial_samples_matlab_ik})
    print('1')

if __name__ == '__main__':
    main()
