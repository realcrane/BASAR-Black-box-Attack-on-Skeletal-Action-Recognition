import torch.nn as nn
import torch
import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss,self).__init__()
        self.a = 1

    def forward(self, x, y, epsilon=1e-20):
        sz = x.size()
        x = self.softmax(x)

        x = torch.clamp(x, epsilon, 1-epsilon)
        log_likelihood = torch.mul(-torch.log(x), y)
        loss = torch.sum(torch.sum(log_likelihood, dim=1)) / sz[0]
        return loss

    def softmax(self, X):
        exps = torch.exp(X)
        temp = torch.div(exps.t(), torch.sum(exps, dim=1))
        return temp.t()
