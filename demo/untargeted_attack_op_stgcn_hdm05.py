import os
import pickle
from datetime import datetime
import numpy as np
from gekko import GEKKO
import shutil
import sys
sys.path.append('../')
import tools.optimization as op
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
sys.path.append('../motion')
from Quaternions import Quaternions
import BVH as BVH
import Animation as Animation
import tools.tools as tools
import argparse


parser = argparse.ArgumentParser(description='BASAR')
parser.add_argument('--w',type=float, default=1, help = 'optimization weights')
parser.add_argument('--w_final',type=float, default=1, help = 'optimization weights')
parser.add_argument('--dataset',default='hdm05')
parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
parser.add_argument('--data_root',default='../data/')
parser.add_argument('--save_root',default='../results/')
parser.add_argument('--model',default='stgcn',help = 'attacked model')
parser.add_argument('--model_root',default='../models/stgcn/hdm05_epoch65_model.pt',help = 'attacked model')
parser.add_argument('--iter',type=int, default=500,help='iterations')
parser.add_argument('--num',type=int, default=5,help='the number of generating adversarial examples')
args = parser.parse_args()

def main(args):
    w = args.w
    w_final = args.w_final
    iterations = args.iter
    dataset = args.dataset
    data_root = args.data_root + dataset + '/'
    if dataset == 'hdm05':
        model = Model(in_channels=3, num_class=65, dropout=0.5, edge_importance_weighting=True,
                      graph_args={})

        core = np.load('../dataprecess/core/hdm05_core_ik.npy')
    path_save = args.save_root + dataset + '/untargeted/' + datetime.now().strftime("%m%d_%H%M%S")
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    phfile = args.model_root
    train_data_path = data_root + 'train_data.npy'
    train_label_path = data_root + 'train_label.pkl'
    val_data_path = data_root + 'val_data.npy'
    val_label_path = data_root + 'val_label.pkl'


    model.load_state_dict(torch.load(phfile))
    model.eval()
    model.cuda()
    # data_size: nX3Xnum_framesX25X2

    T = Feeder(train_data_path, train_label_path, random_choose=False, random_move=False,
               window_size=-1, mmap=True, debug=False)
    V = Feeder(val_data_path, val_label_path, random_choose=False, random_move=False,
               window_size=-1, mmap=True, debug=False)
    Initial_data = torch.utils.data.DataLoader(V, batch_size=1, shuffle=True, drop_last=False)
    Target_data = torch.utils.data.DataLoader(V, batch_size=1, shuffle=True, drop_last=False)
    target_samples = []
    target_samples_post = []
    target_classes = []
    initial_samples = []
    initial_classes = []
    adversarial_samples = []
    ad_eulers = []
    tar_eulers = []
    ad_ik_samples = []
    attack_classes_ik = []
    attack_classes = []
    n_calls_ik = []
    n_calls_list = []
    n_steps_list = []
    diffes = []
    diffes_post = []
    diffes_ik_post = []
    n_target = 0
    w1s = []
    attack_success = 1

    for target_sample_data_old, target_label_old in Target_data:
        target_sample_data_old = target_sample_data_old.float().cuda()
        target_label_old = target_label_old.long().cuda()
        with torch.no_grad():
            target_output = model(target_sample_data_old).cuda()
            _, target_class_old = torch.max(target_output, 1)
        if target_class_old != target_label_old:
            continue
        n_target += 1

        if n_target > args.num:
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
        target_sample_ik,target_sample_euler = kinematics.target_ik(targets_sample_postprocessing,dataset = 'hdm05')
        #tar_post2 = Animation.positions_global(target_sample_ik)
        num_frames = len(target_sample[0,:,0])
        '''#delete
        tar_sample_process2 = Animation.positions_global(target_sample_ik)
        tar_sample2 = torch.from_numpy(tools.normalize(tar_sample_process2[:,1:,:],core))
        target_sample_data[0,:,:,:,0] = tar_sample2
        '''#delete
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
            target_samples.append(deepcopy(target_sample_data[0].cpu().numpy()))
            target_samples_post.append(deepcopy(targets_sample_postprocessing))
            target_classes.append(deepcopy(target_class.cpu().numpy()))
            # initial_sample = np.transpose(initial_sample, [0, 2, 1])  # (3,num_frames,25) (3,25,num_frames)
            n_steps = 0
            n_calls = 0
            epsilon = 0.95
            delta = 0.1
            theta = 0.4
            ik_calls = 0
            ad = False
            adversarial_samples_out = []
            attack_classes_out = []
            diffes_out = []
            diffes_out_post = []
            adversarial_samples_iter = []
            attack_classes_iter = []
            n_calls_list_iter = []
            n_steps_list_iter = []
            while True:
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
                    if d_step >= 300:
                        adversarial_samples.append(deepcopy(adversarial_sample.cpu().numpy()))
                        attack_classes.append(deepcopy(attack_class.cpu().numpy()))
                        diff = torch.mean(get_diff(adversarial_sample.permute(1, 2, 0), target_sample.permute(1, 2, 0)))
                        diffes.append(deepcopy(diff.cpu().numpy()))
                        n_calls_list.append(deepcopy(n_calls))
                        n_steps_list.append(deepcopy(n_steps))
                        n_op_break = 1
                        break
                    for i in torch.arange(5):
                        or_p = orthogonal_perturbation(delta, adversarial_sample, target_sample,dataset)
                        # trial_sample = np.transpose(adversarial_sample + or_p, [0, 2, 1])
                        #trial_sample = torch.from_numpy(np.transpose(adversarial_sample + or_p, [0, 2, 1])).reshape(1, 3, num_frames,25)
                        trial_sample = (adversarial_sample+or_p)#.reshape(1,3,num_frames,25)
                        trial_sample_data[0, :, :, :, 0] = trial_sample
                        trial_output = model(trial_sample_data)
                        _, prediction = torch.max(trial_output, 1)
                        predictions.append(deepcopy(prediction))
                        trial_samples.append(deepcopy(trial_sample))
                    n_calls += 5
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
                                if d_score <= 0.2:
                                    delta_op.append(deepcopy(delta*0.9))
                                elif d_score >= 0.8:
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
                    diff_op.append(deepcopy(diff_trial))
                    '''
                    if dataset == 'hdm05':
                        ad_postprocess = tools.postprocessing(adversarial_sample.cpu().numpy(), core)
                        tar_postprocess = tools.postprocessing(target_sample.cpu().numpy(), core)
                    diff_op.append(deepcopy(diff_trial))
                    '''
                index_op = torch.tensor(diff_op).argmin()
                attack_class = attack_class_op[index_op]
                adversarial_sample = adversarial_samples_op[index_op]
                diff = diff_op[index_op]
                epsilon = epsilon_op[index_op]
                delta = delta_op[index_op]
                if  diff <= 0.1 and delta > 0.2:
                    delta = 0.1
                elif diff <= 0.05 and delta > 0.1:
                    delta = 0.05
                elif diff < 0.025 and delta > 0.05:
                    delta = 0.025

                # argmin
                adversarial_samples_out.append(deepcopy(adversarial_sample))
                attack_classes_out.append(deepcopy(attack_class))
                diffes_out.append(deepcopy(diff))
                n_steps += 1
                print("Mean Squared Error: {}".format(diff))
                print("Calls: {}".format(n_calls))
                print("Attack Class: {}".format(attack_class))
                print("Target Class: {}".format(target_class))
                print("Initial Class:{}".format(initial_class))
                print("Delta: {}".format(delta))
                print('n target:: {}'.format(n_target))
                if n_steps % 250 == 0 and n_steps < iterations:
                    adversarial_sample_postprocess = tools.postprocessing(adversarial_sample.cpu().numpy(), core)
                    adversarial_sample_ik, adversarial_sample_euler = kinematics.ad_ik(target_sample_ik,
                                                                                       adversarial_sample_postprocess)

                    op_sample, op_global = op.op_angle(target_sample_euler, adversarial_sample_euler, adversarial_sample_ik, core,
                                                       w, straetgy=2)
                    epsilon_opt = 0.95
                    while True:
                        trial_sample_data[0, :, :, :, 0] = op_sample
                        op_output = model(trial_sample_data)
                        _, opt_class = torch.max(op_output, 1)
                        n_calls += 1
                        diff = torch.mean(get_diff(op_sample.permute(1, 2, 0), target_sample.permute(1, 2, 0)))
                        print("Mean Squared Error: {}".format(diff))
                        print("Opt Class: {}".format(opt_class))
                        if opt_class != target_class:
                            adversarial_sample = op_sample
                            attack_class = opt_class
                            if delta < 0.01:
                                delta /= 0.2
                            break
                        else:
                            fp_opt = forward_perturbation(epsilon_opt * get_diff(adversarial_sample, op_sample),
                                                          adversarial_sample, op_sample)
                            fp_opt[:,:,0]=0
                            b = torch.isnan(fp_opt)
                            if torch.any(b == True):
                                break
                            op_sample = adversarial_sample + fp_opt  # (1,3,num_frames,25)
                            epsilon_opt *= 0.95
                    adversarial_samples_iter.append(deepcopy(adversarial_sample.cpu().numpy()))
                    attack_classes_iter.append(deepcopy(attack_class.cpu().numpy()))
                    n_calls_list_iter.append(deepcopy(n_calls))
                    n_steps_list_iter.append(deepcopy(n_steps))
                elif n_steps >= iterations:
                    w1 = deepcopy(w_final)
                    adversarial_sample_postprocess = tools.postprocessing(adversarial_sample.cpu().numpy(), core)
                    adversarial_sample_ik, adversarial_sample_euler = kinematics.ad_ik(target_sample_ik,
                                                                                       adversarial_sample_postprocess)
                    epsilon_opt = 0.95
                    while True:
                        op_sample, op_global = op.op_angle(target_sample_euler, adversarial_sample_euler, adversarial_sample_ik,
                                                           core, w1, straetgy=2)
                        trial_sample_data[0, :, :, :, 0] = op_sample
                        op_output = model(trial_sample_data)
                        _, opt_class = torch.max(op_output, 1)
                        n_calls += 1
                        diff = torch.mean(get_diff(op_sample.permute(1, 2, 0), target_sample.permute(1, 2, 0)))
                        print("Mean Squared Error: {}".format(diff))
                        print("Opt Class: {}".format(opt_class))
                        if opt_class != target_class:
                            attack_class = opt_class
                            adversarial_sample = op_sample
                            break
                        else:
                            w1 *= 0.1
                            if w1 < 1e-3:
                                op_sample, op_global = op.op_angle(target_sample_euler, adversarial_sample_euler,
                                                                   adversarial_sample_ik, core, w_final, straetgy=2)
                                while True:
                                    fp_opt = forward_perturbation(epsilon_opt * get_diff(adversarial_sample, op_sample),
                                                                  adversarial_sample, op_sample)
                                    fp_opt[:, :, 0] = 0
                                    b = torch.isnan(fp_opt)
                                    if torch.any(b == True):
                                        break
                                    op_sample = adversarial_sample + fp_opt  # (1,3,num_frames,25)
                                    epsilon_opt *= 0.95
                                    n_calls += 1
                                    diff = torch.mean(get_diff(op_sample.permute(1, 2, 0), target_sample.permute(1, 2, 0)))
                                    trial_sample_data[0, :, :, :, 0] = op_sample
                                    op_output = model(trial_sample_data)
                                    _, opt_class = torch.max(op_output, 1)
                                    print("Mean Squared Error: {}".format(diff))
                                    print("Opt Class: {}".format(opt_class))
                                    if opt_class != target_class:
                                        attack_class = opt_class
                                        adversarial_sample = op_sample
                                        break
                                    elif epsilon_opt < 0.01:
                                        w1 = -1
                                        break
                                    w1 *= 0.1
                                break
                    adv_sample_postprocessing = tools.postprocessing(adversarial_sample.cpu().numpy(), core)
                    ad_ik, adversarial_sample_euler = kinematics.target_ik(adv_sample_postprocessing,
                                                                                 dataset=dataset)
                    ad_eulers.append(deepcopy(adversarial_sample_euler))
                    tar_eulers.append(deepcopy(target_sample_euler))
                    adversarial_samples_iter.append(deepcopy(adversarial_sample.cpu().numpy()))
                    attack_classes_iter.append(deepcopy(attack_class.cpu().numpy()))
                    n_calls_list_iter.append(deepcopy(n_calls))
                    n_steps_list_iter.append(deepcopy(n_steps))
                    w1s.append(deepcopy(w1))
                    diffes.append(deepcopy(diff.cpu().numpy()))
                    ad_ik_samples.append(deepcopy(adversarial_sample.cpu().numpy()))
                    attack_classes_ik.append(deepcopy(attack_class.cpu().numpy()))
                    n_calls_ik.append(deepcopy(n_calls))
                    adversarial_samples_iter = np.array(adversarial_samples_iter)
                    attack_classes_iter = np.array(attack_classes_iter)
                    n_calls_list_iter = np.array(n_calls_list_iter)
                    adversarial_samples.append(deepcopy(adversarial_samples_iter))
                    attack_classes.append(deepcopy(attack_classes_iter))
                    n_calls_list.append(deepcopy(n_calls_list_iter))
                    np.save(os.path.join(path_save, 'target_classes.npy'), np.array(target_classes))
                    np.save(os.path.join(path_save, 'initial_classes.npy'), np.array(initial_classes))
                    np.save(os.path.join(path_save, 'adversarial_ik_samples.npy'), np.array(ad_ik_samples))
                    np.save(os.path.join(path_save, 'attack_classes_ik.npy'), np.array(attack_classes_ik))
                    np.save(os.path.join(path_save, 'initial_samples.npy'), np.array(initial_samples))
                    np.save(os.path.join(path_save, 'target_samples.npy'), np.array(target_samples))
                    np.savetxt(os.path.join(path_save, 'diffes.txt'), np.array(diffes))
                    break


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
