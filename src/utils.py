# coding=utf-8
import shutil

import gpustat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.optimizer import Optimizer


class ReduceLROnPlateau(object):

    def __init__(self, optimizer, mode='min', factor=0.5, patience=5,
                 verbose=True, epsilon=1E-3, min_lr=0.):

        if factor <= 0.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor <= 0.0')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self.reset()

    def reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            raise RuntimeError(
                'Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min':
            self.monitor_op = lambda a, b: a < (b - self.epsilon)
            self.best = 1E12
        else:
            self.monitor_op = lambda a, b: a > (b + self.epsilon)
            self.best = -1E12
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1E-4

    def step(self, metric, epoch):
        if self.monitor_op(metric, self.best):
            self.best = metric
            self.wait = 0

        elif self.wait >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = float(param_group['lr'])
                if old_lr > (self.min_lr + self.lr_epsilon):
                    new_lr = old_lr * self.factor
                    param_group['lr'] = max(new_lr, self.min_lr)
                    if self.verbose:
                        print('Reducing learning rate to %s.' % new_lr)
                    self.wait = 0
        else:
            self.wait += 1


def categorical_crossentropy(y_pred, y_true, batch_average=True, timestep_average=True):
    # scale preds so that the class probas of each sample sum to 1
    cumsum = torch.sum(y_pred, dim=-1).repeat(1, 1, y_pred.size()[
        -1])  # need to repeat until we have keepdim from master
    y_pred /= cumsum
    # manual computation of crossentropy
    epsilon = 1E-7
    output = F.hardtanh(y_pred, min_val=epsilon, max_val=1. - epsilon)
    loss = -torch.sum(y_true.detach() * torch.log(output))

    if batch_average:
        loss /= y_pred.size()[0]
    if timestep_average:
        loss /= y_pred.size()[1]
    return loss


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
    for m in m.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data = init.xavier_uniform(m.weight.data, gain=1)
        elif isinstance(m, nn.Linear):
            m.weight.data = init.xavier_uniform(m.weight.data, gain=1)
            m.bias.data.zero_()
        elif isinstance(m, nn.GRU):
            for i in range(m.num_layers):
                ih = getattr(m, "weight_ih_l{}".format(i))
                hh = getattr(m, "weight_hh_l{}".format(i))
                b_ih = getattr(m, "bias_hh_l{}".format(i))
                b_hh = getattr(m, "bias_hh_l{}".format(i))
                ih.data = init.xavier_uniform(ih.data, gain=init.calculate_gain("tanh"))
                hh.data = init.xavier_uniform(hh.data, gain=init.calculate_gain("tanh"))
                b_ih.data.zero_()
                b_hh.data.zero_()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def show_memusage(mystring="", device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}{}/{}".format(mystring, item["memory.used"], item["memory.total"]))


def get_memusage(device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    return gpu_stats.jsonify()["gpus"][device]


def register_nan_checks(model, grad=True, output=False):
    # a != a returns a byte tensor with 1 for each NaN. This way we can check the NaN on the GPU as well
    def check_grad(module, grad_input, grad_output):
        # print(module) you can add this to see that the hook is called
        for grad in grad_output:
            if grad is not None:
                if ((torch.sum((grad.data != grad.data).view(-1), 0)[0]) > 0):
                    model.nangrad = True
                    print('NaN gradient in ' + type(module).__name__)
        # if any((torch.sum((grad.data != grad.data).view(-1), 0)[0]) > 0 for grad in grad_output if grad is not None):

    def check_output(module, input, output):
        # print(module) you can add this to see that the hook is called
        if any(((torch.sum((out.data != out.data).view(-1), 0)[0]) > 0 for out in output if out is not None)):
            model.nanout = True
            print('NaN output in ' + type(module).__name__)

    if grad:
        model.apply(lambda module: module.register_backward_hook(check_grad))
    if output:
        model.apply(lambda module: module.register_forward_hook(check_output))


def acc(o: torch.FloatTensor, t: torch.FloatTensor):
    _, o_maxes = torch.max(o, dim=2)
    _, t_maxes = torch.max(t, dim=2)
    t_maxes = t_maxes.type_as(o_maxes)
    return torch.sum(o_maxes.view(-1) == t_maxes.view(-1)) / o_maxes.view(-1).size(0)


def samples_multidimensional_ball(center, radius=1., num_data=10, dim=56):
    # type: (int, numpy.ndarray, int) -> object

    # generate a number of points on a unit hypersphere
    x = np.random.normal(size=(num_data, dim))

    # generate a random number of radii
    r = np.power(np.random.uniform(0.0, radius ** dim, num_data), 1 / float(dim))

    # scale points with the radius as a scaling factor
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    x *= r[:, np.newaxis]
    # move the points towards the center
    x += center
    return x


def samples_multidimensional_sphere(center, radius=1., num_data=10, dim=56):
    # type: (int, numpy.ndarray, int) -> object

    x = np.random.normal(size=(num_data, dim))
    r = np.array([radius] * num_data)
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    x *= r[:, np.newaxis]
    x += center
    return x
