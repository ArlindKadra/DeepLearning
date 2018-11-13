from torch.optim.optimizer import Optimizer
import numpy as np
import math
import logging


class CosineScheduler(object):

    def __init__(self, optimizer, nr_epochs,
                 weight_decay=False, restart=False):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        self.restart = restart
        self.weight_decay = weight_decay
        self.nr_epochs = nr_epochs
        self.anneal_max_epoch = math.ceil(1 / 10 * nr_epochs)
        self.anneal_multiply = 2
        self.anneal_epoch = 0

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            if weight_decay:
                group.setdefault('initial_weight_decay',
                                 group['weight_decay'])

    # Starting from epoch 0
    def step(self, epoch):

        # TODO refactor the usage of epoch
        # the cosine annealing case
        if self.restart:
            if self.anneal_epoch > self.anneal_max_epoch:
                logger = logging.getLogger(__name__)
                logger.error("Something went wrong, epochs "
                             "> max epochs in restart")
                raise ValueError("Something went wrong, epochs "
                                 "> max epochs in restart")

            if self.anneal_epoch == self.anneal_max_epoch:
                self.anneal_epoch = 0
                self.nr_epochs -= self.anneal_max_epoch
                decay = 1
                if self.nr_epochs >= self.anneal_max_epoch * self.anneal_multiply:
                    self.anneal_max_epoch = self.anneal_max_epoch * \
                                            self.anneal_multiply
                else:
                    self.anneal_max_epoch = self.nr_epochs
            else:
                decay = 0.5 * (1.0 + np.cos(
                    np.pi * (self.anneal_epoch / self.anneal_max_epoch)
                ))
            self.anneal_epoch += 1
        # Cosine Decay
        else:
            decay = 0.5 * (1.0 + np.cos(
                np.pi * (epoch / self.nr_epochs)
            ))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * decay
            if self.weight_decay:
                param_group['weight_decay'] = param_group['initial_weight_decay'] * decay


class ExponentialScheduler(object):

    def __init__(self, optimizer, nr_epochs, final_fraction, weight_decay=False):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.beta = nr_epochs / -np.log(final_fraction)

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            if weight_decay:
                group.setdefault('initial_weight_decay', group['weight_decay'])

    # Starting from epoch 0
    def step(self, epoch):

        decay = np.exp(-epoch / self.beta)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * decay
            if self.weight_decay:
                param_group['weight_decay'] = param_group['initial_weight_decay'] * decay


class ScheduledOptimizer(object):

    def __init__(self,
                 optimizer,
                 nr_epochs,
                 weight_decay,
                 scheduler=None,
                 final_fraction=0.1
                 ):

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
                    nr_epochs,
                    final_fraction,
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
