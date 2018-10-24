import time
import os
import argparse
import logging
import math

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import model
import config
from utilities import plot, log
import optim.hpbandster
from utilities import data, regularization
from optim.adamw import AdamW
from optim.sgdw import SGDW
from optim.lr_scheduler import ScheduledOptimizer
from models.fcresnet import FcResNet
from models.fcnet import FcNet


def main():

    parser = argparse.ArgumentParser(description='Fully connected residual network')
    parser.add_argument('--num_workers', help='Number of hyperband workers.', default=4, type=int)
    parser.add_argument('--num_iterations', help='Number of hyperband iterations.', default=4, type=int)
    parser.add_argument('--run_id', help='unique id to identify the HPB run.', default='HPB_example_2', type=str)
    parser.add_argument('--array_id', help='SGE array id to tread one job array as a HPB run.', default=1, type=int)
    parser.add_argument('--working_dir', help='working directory to store live data.', default='.', type=str)
    parser.add_argument('--nic_name', help='name of the Network Interface Card.', default='lo', type=str)
    parser.add_argument('--network_type', help='network to be used for the task.', default='fcresnet', type=str)
    parser.add_argument('--cluster_workload', help='Workload management package.', default='slurm', type=str)
    parser.add_argument('--cross_validation', help='Cross-Validation flag', default=False, type=bool)
    parser.add_argument(
        '--predictive_measure',
        help='Measure which will be passed to the hyperparameter '
             'optimization procedure on the end of the run',
        default='loss',
        type=str
    )
    parser.add_argument('--task_id', help='Task id so that the dataset can be retrieved from OpenML.',
                        default=3, type=int)

    args = parser.parse_args()
    config.network_type = args.network_type
    config.cross_validation = args.cross_validation
    # initialize logging
    logger = logging.getLogger(__name__)

    # TODO put verbose into configuration file
    verbose = False

    # In MOAB, the run_id also has the array_id. This
    # conflicts with the way BOHB stores the nameserver
    # config
    if args.cluster_workload == 'slurm':
        run_id = args.run_id
    else:
        run_id = args.run_id[:-3]

    log.setup_logging(run_id + "_" + str(args.array_id), logging.DEBUG if verbose else logging.INFO)
    logger.info('DeepResNet Experiment started')
    model.Loader(args.task_id)
    working_dir = os.path.join(args.working_dir, 'task_%i' % model.get_task_id(), args.network_type)
    start_time = time.time()
    optim.hpbandster.Master(args.num_workers, args.num_iterations, run_id, args.array_id, working_dir, args.nic_name, args.network_type)
    end_time = time.time()
    duration = (end_time - start_time) / 60
    
    if args.array_id == 1:

        plot.test_loss_over_budgets(working_dir)
        plot.best_conf_val_loss(working_dir)
        plot.plot_curves(working_dir)
        plot.plot_rank_correlations(working_dir)
        log.general_info(working_dir, duration)


def train(config, network, num_epochs, x, y, set_indices):

    logger = logging.getLogger(__name__)
    dataset_categorical = model._categorical
    # number of dataset classes
    nr_classes = max(y) + 1

    # unpack set indices
    train_indices = set_indices[0]
    val_indices = set_indices[1]
    test_indices = set_indices[2]

    # calculate mean and std for train set
    x_train = x[train_indices]
    mean, std = data.calculate_stat(x_train)
    x = data.feature_normalization(x, mean, std, dataset_categorical)

    # Deal with categorical attributes
    if True in dataset_categorical:

        enc = OneHotEncoder(categorical_features=dataset_categorical, dtype=np.float32)
        x = enc.fit_transform(x).todense()

    x_train = x[train_indices]
    x_val = x[val_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    # Get the batch size
    batch_size = config["batch_size"]

    device = torch.device("cuda"
                          if torch.cuda.is_available() else "cpu")

    if network == 'fcresnet':
        network = FcResNet(
            config,
            x_train.shape[1],
            nr_classes).to(device)
    elif network == 'fcnet':
        network = FcNet(
            config,
            x_train.shape[1],
            nr_classes).to(device)
    else:
        raise ValueError('Only fcresnet and fcnet are allowed'
                         'as values')

    # Calculate the number of parameters for the network
    total_params = sum(p.numel() for
                       p in network.parameters() if p.requires_grad)
    logger.info("Number of network parameters %d", total_params)

    criterion = nn.CrossEntropyLoss()
    weight_decay = 0
    if 'weight_decay' in config:
        weight_decay = config['weight_decay']
    # Optimizer to be used
    if config['optimizer'] == 'SGDW':
        # Although l2_regularization is being passed as weight
        # decay, it is ~ the same thing. What is done, is
        # a decoupling of the regularization and learning rate.
        optimizer = SGDW(
            network.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=weight_decay,
            nesterov=True
        )
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=weight_decay,
            nesterov=True
        )
    elif config['optimizer'] == 'AdamW':
        optimizer = AdamW(
            network.parameters(),
            lr=config['learning_rate'],
            weight_decay=weight_decay
        )
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config['learning_rate'],
            weight_decay=weight_decay
        )
    else:
        logger.error("Unexpected optimizer value, "
                     "legal values are: SGD, SGDW"
                     ", Adam and AdamW")
        raise ValueError("Unexpected optimizer value")

    scheduled_optimizer = ScheduledOptimizer(
        optimizer,
        num_epochs,
        True if weight_decay != 0 else False,
        config['decay_type']
    )
    logger.info('FcResNet started training')

    # array to save the validation loss for each epoch
    network_val_loss = []
    # array to save the validation accuracy for each epoch
    network_val_accuracy = []
    # array to save the training loss for each epoch
    network_train_loss = []

    x_val = torch.from_numpy(x_val).float()
    x_val.requires_grad_(False)
    y_val = torch.from_numpy(y_val)
    y_val.requires_grad_(False)
    x_val, y_val = x_val.to(device), y_val.to(device)

    # loop over the dataset according to the number of epochs
    for epoch in range(0, num_epochs):

        running_loss = 0.0
        nr_batches = 0
        # train the network
        network.train()

        for i in range(0, (x_train.shape[0] - batch_size), batch_size):

            indices = np.arange(i, i + batch_size)
            shuffled_indices = np.random.permutation(indices)
            lam = 1
            # Check if mixup is active
            if 'mixout_alpha' in config:
                mixout_alpha = config['mixout_alpha']
                lam = np.random.beta(mixout_alpha, mixout_alpha)

            x = lam * x_train[indices] + \
                (1 - lam) * x_train[shuffled_indices]

            targets_a = y_train[indices]
            targets_b = y_train[shuffled_indices]
            targets_a = torch.from_numpy(targets_a).long()
            targets_b = torch.from_numpy(targets_b).long()
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)
            loss_function = regularization.mixup_criterion(targets_a, targets_b, lam)
            x = torch.from_numpy(x).float()
            x = x.to(device)

            # zero the gradient buffers
            scheduled_optimizer.zero_grad()
            output = network(x)

            # stop training if we have NaN values in the output
            if data.contains_nan((output.cpu()).data.numpy()):
                # TODO switch to logger exception
                logger.error('Output contains NaN values')
                raise ValueError("NaN value in output")

            loss = loss_function(criterion, output)

            loss.backward()
            running_loss += loss.item()
            nr_batches += 1

        # Using validation data
        network.eval()

        correct = 0
        total = 0
        outputs = network(x_val)
        val_loss = criterion(outputs, y_val).item()

        # bad configuration, stop training
        # add -1 as the loss for the validation
        # and test
        if np.isnan(val_loss):
            network_val_loss.append(math.inf)
            return {
                'test': (math.inf, 0),
                'validation': (network_val_loss, 0),
                'train': network_train_loss
            }

        _, predicted = torch.max(outputs.data, 1)
        total += y_val.size(0)
        correct += ((predicted == y_val).sum()).item()
        val_accuracy = 100 * correct / total

        network_train_loss.append(running_loss / nr_batches)
        network_val_loss.append(val_loss)
        network_val_accuracy.append(val_accuracy)
        logger.info(
            'Epoch %d, Train loss: %.3f, '
            'Validation loss: %.3f, accuracy %.3f',
            epoch + 1,
            running_loss / nr_batches,
            val_loss,
            val_accuracy
        )
        logger.info(
            'Learning rate: %.3f',
            scheduled_optimizer.get_learning_rate()
        )
        logger.info(
            'Weight decay: %.3f',
            scheduled_optimizer.get_weight_decay()
        )

        scheduled_optimizer.step(epoch)

    with torch.no_grad():
        correct = 0
        total = 0
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).long()
        x_test, y_test = x_test.to(device), y_test.to(device)
        network.eval()
        outputs = network(x_test)
        test_loss = criterion(outputs, y_test)
        _, predicted = torch.max(outputs.data, 1)
        total += y_test.size(0)
        correct += ((predicted == y_test).sum()).item()
        accuracy = 100 * correct / total
    logger.info('Test loss: %.3f, accuracy of the network: %.3f %%', test_loss.item(), accuracy)
    output_information = {
        'test': (test_loss.item(), accuracy),
        'validation': (network_val_loss, network_val_accuracy),
        'train': network_train_loss
    }
    return output_information


if __name__ == '__main__':
    main()
