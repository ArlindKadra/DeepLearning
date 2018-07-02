from torch.optim.optimizer import Optimizer
import numpy as np

# https://github.com/PatrykChrabaszcz/DeepLearning_EEG/blob/85ffdde2ac77d14b42fc40919fc4ed478f75aa17/src/deep_learning/pytorch/optimizer.py
class CosineScheduler:
    def __init__(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            group.setdefault('initial_weight_decay', group['weight_decay'])

    # Starting from epoch 0
    def step(self, progress):
        assert 0.0 <= progress <= 1.0
        decay = 0.5 * (1.0 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * decay
            param_group['weight_decay'] = param_group['initial_weight_decay'] * decay


# Set scheduler to CosineScheduler if you want to use cosine annealing
class ScheduledOptimizer(object):
    def __init__(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = scheduler(optimizer)
        else:
            self.scheduler = None

    # Use progress to derive cosine phase
    def step(self, progress):
        if self.scheduler is not None:
            self.scheduler.step(progress)
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
