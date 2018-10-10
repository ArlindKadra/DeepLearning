from torch.optim.optimizer import Optimizer
import numpy as np
import math
import logging

class CosineScheduler:

    def __init__(self, optimizer, nr_epochs,
                 weight_decay= False, restart=False):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        self.restart = restart
        self.weight_decay = weight_decay
        self.nr_epochs = nr_epochs
        self.anneal_max_epoch = int(1 / 3 * nr_epochs)
        self.anneal_multiply = 2

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            if weight_decay:
                group.setdefault('initial_weight_decay',
                                 group['weight_decay'])

    # Starting from epoch 0
    def step(self, epoch):

        # the cosine annealing case
        if self.restart:
            if epoch > self.anneal_max_epoch:
                logger = logging.getLogger(__name__)
                logger.error("Something went wrong, epochs "
                             "> max epochs in restart")
                raise ValueError("Something went wrong, epochs "
                                 "> max epochs in restart")

            if epoch == self.anneal_max_epoch:
                decay = 1
                self.anneal_max_epoch = self.anneal_max_epoch * \
                                            self.anneal_multiply
            else:
                decay = 0.5 * (1.0 + np.cos(
                    np.pi * (epoch / self.anneal_max_epoch)
                ))
        # Cosine Decay
        else:
            decay = 0.5 * (1.0 + np.cos(
                np.pi * (epoch / self.nr_epochs)
            ))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * decay
            if self.weight_decay:
                param_group['weight_decay'] = param_group['initial_weight_decay'] * decay


class ExponentialScheduler:

    def __init__(self, optimizer, weight_decay=False):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.gamma = math.e

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            if weight_decay:
                group.setdefault('initial_weight_decay', group['weight_decay'])

    # Starting from epoch 0
    def step(self, epoch):

        decay = self.gamma ** (-epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * decay
            if self.weight_decay:
                param_group['weight_decay'] = param_group['initial_weight_decay'] * decay


class ScheduledOptimizer(object):

    def __init__(self, optimizer, nr_epochs, weight_decay, scheduler=None):

        self.optimizer = optimizer
        if scheduler is not None:
            if scheduler == 'cosine_annealing':
                self.scheduler = CosineScheduler(
                    optimizer,
                    nr_epochs,
                    restart=True,
                    weight_decay=weight_decay
                )
            elif scheduler == 'cosine_decay':
                self.scheduler = CosineScheduler(
                    optimizer,
                    nr_epochs,
                    restart=False,
                    weight_decay=weight_decay
                )
            elif scheduler == 'exponential_decay':
                self.scheduler = ExponentialScheduler(
                    optimizer,
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(
                    'The schedule can only be cosine_'
                    'annealing, cosine_decay or exponential_decay.'
                )
        else:
            self.scheduler = None

    def step(self, epoch):

        if self.scheduler is not None:
            self.scheduler.step(epoch)
        self.optimizer.step()

    def state_dict(self):

        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):

        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):

        self.optimizer.zero_grad()

    def get_learning_rate(self):

        return self.optimizer.param_groups[0]['lr']

    def get_weight_decay(self):

        return self.optimizer.param_groups[0]['weight_decay']
