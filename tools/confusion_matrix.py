from sklearn.metrics import confusion_matrix
import os
import pickle
import numpy as np
import shutil
from copy import deepcopy
import torch
import tools.tools as tools
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

def confusionMat(motions,orMotions,plabels,tlabels, folder = './', fileName = 'confuseMat'):
    '''
    data = np.load(file)

    motions = data['clips']
    orMotions = data['oriClips']
    plabels = data['classes']
    tlabels = data['tclasses']
    '''
    hitIndices = []

    for i in range(0, len(tlabels)):
        if plabels[i] != tlabels[i]:
            hitIndices.append(i)

    hitIndices = np.array(hitIndices)

    tclasses = tlabels[hitIndices]
    pclasses = plabels[hitIndices]


    #if os.path.isfile(folder + 'classes.txt'):
    #    with open(folder + 'classes.txt') as f:
    #        content = f.readlines()
    #        labels = [x.strip() for x in content]

    #labelsInds = np.array(np.union1d(tclasses, pclasses))


    #labels = [labels[i] for i in labelsInds]

    #mat = confusion_matrix(tclasses, pclasses)


    #f = plt.figure(figsize=(19, 15))
    #plt.matshow(mat, fignum=f.number)
    #plt.xticks(range(mat.shape[1]), fontsize=14, rotation=45)
    #plt.yticks(range(mat.shape[1]), fontsize=14)
    #cb = plt.colorbar()
    #cb.ax.tick_params(labelsize=14)
    ##plt.title('Confusion Matrix on ', fontsize=16);
    #plt.savefig(fname= folder + '/confusion.png')
    #plt.close(f)

    if os.path.isfile(folder + 'classes.txt'):
        with open(folder + 'classes.txt') as f:
            content = f.readlines()
            class_names = [x.strip() for x in content]

    class_names = [i for i in range(len(class_names))]

    np.set_printoptions(precision=2)

    ## Plot non-normalized confusion matrix
    #f, ax = plot_confusion_matrix(tclasses, pclasses, classes=class_names,
    #                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    f, ax = plot_confusion_matrix(tclasses, pclasses, classes=class_names, normalize=False,
                          )

    plt.savefig(fname= folder + '/' + fileName + '.png')
    plt.close(f)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'


    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm[29,46]=1
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = [classes[i] for i in np.array(unique_labels(y_true, y_pred), dtype=np.int)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=(19, 15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.xlabel('Predicted label', fontsize=18)
    plt.ylabel('True label', fontsize=18)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #fmt = '.2f' if normalize else 'd'
    #thresh = cm.max() / 2.
    #for i in range(cm.shape[0]):
    #    for j in range(cm.shape[1]):
    #        ax.text(j, i, format(cm[i, j], fmt),
    #                ha="center", va="center",
    #                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax
