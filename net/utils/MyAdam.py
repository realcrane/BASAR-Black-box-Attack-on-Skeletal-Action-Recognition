import torch
import numpy as np

class MyAdam:
    def __init__(self, lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False, initial_decay=0):

        # Arguments
        # lr: float >= 0. Learning rate.
        # beta_1: float, 0 < beta < 1. Generally close to 1.
        # beta_2: float, 0 < beta < 1. Generally close to 1.
        # epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        # decay: float >= 0. Learning rate decay over each update.
        # amsgrad: boolean. Whether to apply the AMSGrad variant of this
        #    algorithm from the paper "On the Convergence of Adam and
        #    Beyond".

        iteration = 0.0
        iteration = np.array(iteration)
        lr = np.array(lr)
        beta_1 = np.array(beta_1)
        beta_2 = np.array(beta_2)
        epsilon = np.array(epsilon)
        decay = np.array(decay)
        initial_decay = np.array(initial_decay)
        self.iteration = torch.from_numpy(iteration).type(torch.FloatTensor)
        self.learningRate = torch.from_numpy(lr).type(torch.FloatTensor)
        self.beta_1 = torch.from_numpy(beta_1).type(torch.FloatTensor)
        self.beta_2 = torch.from_numpy(beta_2).type(torch.FloatTensor)
        self.epsilon = torch.from_numpy(epsilon).type(torch.FloatTensor)
        self.decay = torch.from_numpy(decay).type(torch.FloatTensor)
        self.amsgrad = amsgrad
        self.initial_decay = torch.from_numpy(initial_decay).type(torch.FloatTensor)

    def get_updates(self, grads, params):

        N, C, T, V, M = params.size()
        params = params.permute(0, 4, 2, 3, 1).contiguous()
        params = params.view(N * M, T, V, C)

        rets = torch.zeros(params.shape).cuda()

        lr = self.learningRate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * self.iteration))

        t = self.iteration + 1
        lr_t = lr * (torch.sqrt(1. - torch.pow(self.beta_2, t)) /
                     (1. - torch.pow(self.beta_1, t)))
        lr_t = lr_t.cuda()

        epsilon = self.epsilon.cuda()

        ms = torch.zeros(params.shape).cuda()
        vs = torch.zeros(params.shape).cuda()

        if self.amsgrad:
            vhats = torch.zeros(params.shape).cuda()
        else:
            vhats = torch.zeros(params.shape).cuda()

        for i in range(0, rets.shape[0]):
            p = params[i]
            g = grads[i]
            m = ms[i]
            v = vs[i]
            vhat = vhats[i]

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * torch.mul(g, g)
            if self.amsgrad:
                vhat_t = torch.max(vhat, v_t)
                p_t = p - lr_t * m_t / (torch.sqrt(vhat_t) + epsilon)
                vhat = vhat_t

            else:
                p_t = p - lr_t * m_t / (torch.sqrt(v_t) + epsilon)

            rets[i] = p_t

        return rets

    def zero_grad(self, model):
        """Sets gradients of all model parameters to zero."""
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()