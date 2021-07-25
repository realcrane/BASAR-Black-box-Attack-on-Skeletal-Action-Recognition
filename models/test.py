import os

import torch.nn.parallel
import torch.optim

from net.twosgcn.agcn import Model
from utils.opts import parser
from collections import OrderedDict

import torch
import numpy as np

data = np.load('../data/mhad/stgcn/val_ad_data.npy')

best_acc = 0

def main():
    global args, best_acc
    args = parser.parse_args()

    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    gpuid = [str(i) for i in args.gpus]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpuid)

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    model = Model(in_channels=args.k, num_class=11, num_person=1, graph='selfdefine', graph_args={})
    print(model)

    model=load_weights(model, weights_path= './twosgcn/mhad_agcn_joint-8.pt')
    torch.save( {'state_dict': model.state_dict()}, './twosgcn/mhad_agcn_joint-8.pth.tar')


def load_weights(model, weights_path, ignore_weights=None):
    if ignore_weights is None:
        ignore_weights = []
    if isinstance(ignore_weights, str):
        ignore_weights = [ignore_weights]

    print('Load weights from {}.'.format(weights_path))
    weights = torch.load(weights_path)
    weights = OrderedDict([[k.split('module.')[-1],
                            v.cpu()] for k, v in weights.items()])

    # filter weights
    for i in ignore_weights:
        ignore_name = list()
        for w in weights:
            if w.find(i) == 0:
                ignore_name.append(w)
        for n in ignore_name:
            weights.pop(n)
            print('Filter [{}] remove weights [{}].'.format(i, n))

    for w in weights:
        print('Load weights [{}].'.format(w))

    try:
        model.load_state_dict(weights)
    except (KeyError, RuntimeError):
        state = model.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        for d in diff:
            print('Can not find weights [{}].'.format(d))
        state.update(weights)
        model.load_state_dict(state)
    return model

if __name__ == '__main__':
    main()