import numpy as np
import pdb
import os
import shutil
import glob

path = '../us/'
dataset = 'ntu60'
core = np.load('./core/hdm05_test_preprocess_core.npz')

std = core['Xstd'].reshape(len(core['Xstd'][0]))
mean = core['Xmean'].reshape(len(core['Xmean'][0]))




total_motions =[]
total_plabels =[]
total_orMotions =[]
total_tlabels =[]

folderList= os.listdir(path)
folderList.sort()
for folderName in folderList:
    dataFolder = os.path.join(path, folderName)
    fileList = glob.glob(dataFolder + '/*.npz')

    for filePath in fileList:
        if '_rets.npz' in filePath:
            print(filePath)
            continue

        dirName = filePath[0:-4] + '/'

        if os.path.exists(dirName):
            shutil.rmtree(dirName)


        os.mkdir(dirName)
        data = np.load(filePath)


        motions = data['clips']
        orMotions = data['oriClips']
        plabels = data['classes']
        tlabels = data['tclasses']

        if dataset == 'mhad':
            core = np.load('./core/mhad_test_preprocess_core.npz')

            std = core['Xstd'].reshape(len(core['Xstd'][0]))
            mean = core['Xmean'].reshape(len(core['Xmean'][0]))

            for i in range(len(motions)):
                motions[i] = motions[i] * std + mean
                orMotions[i] = orMotions[i] * std + mean

        elif dataset == 'hdm05':
            core = np.load('./core/hdm05_test_preprocess_core.npz')

            std = core['Xstd'].reshape(len(core['Xstd'][0]))
            mean = core['Xmean'].reshape(len(core['Xmean'][0]))

            for i in range(len(motions)):
                motions[i] = motions[i] * std + mean
                orMotions[i] = orMotions[i] * std + mean

        else:
            core = np.load('./core/ntu_test_preprocess_core.npz')

            std = core['Xstd'].reshape(len(core['Xstd'][0]))
            mean = core['Xmean'].reshape(len(core['Xmean'][0]))

            for i in range(len(motions)):
                motions[i] = motions[i] * std + mean
                orMotions[i] = orMotions[i] * std + mean

        np.save(dirName + 'ad_motions.npy', motions)
        np.savetxt(dirName + 'ad_plabels.txt', plabels)
        np.save(dirName + 'ori_motions.npy', orMotions)
        np.savetxt(dirName + 'tlabels.txt', tlabels)
        '''
        sz = motions.shape
        if dataset == 'ntu':
                motions = np.reshape(motions, [-1, 2, sz[1], 75])
                orMotions = np.reshape(orMotions, [-1, 2, sz[1], 75])
                motions = motions[:, 0, :, :]
                motions = np.reshape(motions, [-1, sz[1], 75])

                orMotions = orMotions[:, 0, :, :]
                orMotions = np.reshape(orMotions, [-1, sz[1], 75])

        total_motions.append(motions)
        total_plabels.append(plabels)
        total_orMotions.append(orMotions)
        total_tlabels.append(tlabels)
total_motions = np.concatenate(total_motions, axis = 0)
total_orMotions = np.concatenate(total_orMotions, axis = 0)
total_plabels = np.concatenate(total_plabels, axis = 0)
total_tlabels = np.concatenate(total_tlabels, axis = 0)

np.save('./ntu_' + 'ad_motions.npy', total_motions)
np.savetxt('./ntu_' + 'ad_plabels.txt', total_plabels)
np.save('./ntu_' + 'ori_motions.npy', total_orMotions)
np.savetxt('./ntu_' + 'tlabels.txt', total_tlabels)

'''



