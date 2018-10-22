import math

import torch
from torch.autograd import Function
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import models.fcresnet


def cross_validation(nr_epochs, x, y, config, nr_folds=10):
    """Use cross validation to train the network.

    Args:
        nr_epochs: Number of epochs to train the network on.
        x: Input.
        y: Labels.
        config: ConfigSpace configuration
        nr_folds: Number of cross-validation folds.

    Returns:
        A float value which shows the average accuracy
        achieved while training with cross-validation.
    """

    val_loss_epochs = np.zeros(nr_epochs)
    val_accuracy = 0
    test_loss = 0
    test_accuracy = 0

    # Shuffle data before, otherwise the results on some tasks were confusing.
    # Validation had similiar loss to the training data while test data had a very high one.
    # np.random.shuffle(x)

    kf = KFold(n_splits=nr_folds, shuffle=True, random_state=11)

    for train_indices, test_indices in kf.split(x):
        x_train, y_train = x[train_indices], y[train_indices]
        x_train, x_validation, \
            y_train, y_validation = \
            train_test_split(x_train, y_train, test_size=1 / (nr_folds - 1), random_state=69)
        x_test, y_test = x[test_indices], y[test_indices]
        output = models.fcresnet.train(config, nr_epochs, x_train, y_train, x_validation, y_validation, x_test, y_test)

        # check last element if it is not inf
        # otherwise there is no point in running
        # cross validation on a bad config
        if (output['validation'][0])[-1] is math.inf:
            return {
                'train_loss': list(output['train']),
                'val_loss': list(output['validation'][0]),
                'val_accuracy': 0,
                'test_loss': math.inf,
                'test_accuracy': 0
        }

        train_loss_epochs = np.add(train_loss_epochs, output['train'])
        val_loss_epochs = np.add(val_loss_epochs, output['validation'][0])
        val_accuracy += output['validation'][1]
        test_loss += output['test'][0]
        test_accuracy += output['test'][1]

    # average the values over the folds
    train_loss_epochs = train_loss_epochs / nr_folds
    val_loss_epochs = val_loss_epochs / nr_folds
    val_accuracy = val_accuracy / nr_folds
    test_loss = test_loss / nr_folds
    test_accuracy = test_accuracy / nr_folds
    result = {
        'train_loss': list(train_loss_epochs),
        'val_loss': list(val_loss_epochs),
        'val_accuracy': val_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    return result


def mixup_criterion(y_a, y_b, lam):

    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Original code from:
# https://github.com/hysts/pytorch_image_classification/blob/master/functions/shake_shake_function.py
class ShakeFunction(Function):

    @staticmethod
    def forward(context, x1, x2, alpha, beta):
        context.save_for_backward(x1, x2, alpha, beta)

        y = x1 * alpha + x2 * (1 - alpha)
        return y

    @staticmethod
    def backward(context, grad_output):
        x1, x2, alpha, beta = context.saved_variables
        grad_x1 = grad_x2 = grad_alpha = grad_beta = None

        if context.needs_input_grad[0]:
            grad_x1 = grad_output * beta
        if context.needs_input_grad[1]:
            grad_x2 = grad_output * (1 - beta)

        return grad_x1, grad_x2, grad_alpha, grad_beta


shake_function = ShakeFunction.apply


def get_alpha_beta(batch_size, shake_config, is_cuda):

    """

    :param batch_size: Number of examples in the batch.

    :param shake_config: Shake shake configuration. Is composed of three values:
        "shake_forward": true,
        "shake_backward": true,
        "shake_image": true

    :param is_cuda: if the network is being run on cuda.

    :return:
        The 2 constants alpha and beta.
    """
    forward_shake, backward_shake, shake_image = shake_config

    # TODO Current implementation does not support shake even

    if forward_shake and not shake_image:
        alpha = torch.rand(1)
    elif forward_shake and shake_image:
        alpha = torch.rand(batch_size).view(batch_size, 1)
    else:
        alpha = torch.FloatTensor([0.5])

    if backward_shake and not shake_image:
        beta = torch.rand(1)
    elif backward_shake and shake_image:
        beta = torch.rand(batch_size).view(batch_size, 1)
    else:
        beta = torch.FloatTensor([0.5])

    if is_cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()

    return alpha, beta
