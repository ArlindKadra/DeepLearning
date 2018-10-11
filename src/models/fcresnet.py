from utilities import data
from optim.adamw import AdamW
from optim.sgdw import SGDW
from optim.lr_scheduler import ScheduledOptimizer

import utilities.regularization
import logging
import ConfigSpace
import torch
import torch.nn as nn
import numpy as np


# TODO Max number of layers and res blocks should be read from the config file
def get_config_space(max_num_layers=2, max_num_res_blocks=14):

    # Config
    optimizers = ['SGDW', 'AdamW', 'SGD', 'Adam']
    block_types = ['BasicRes', 'PreRes']
    decay_scheduler = [
        'cosine_annealing',
        'cosine_decay',
        'exponential_decay'
    ]
    include_hyperparameter = ['Yes', 'No']

    cs = ConfigSpace.ConfigurationSpace()

    # Architecture parameters
    num_layers = ConfigSpace.UniformIntegerHyperparameter(
        "num_layers",
        lower=1,
        upper=max_num_layers,
        default_value=2)
    num_res_blocks = ConfigSpace.UniformIntegerHyperparameter(
        "num_res_blocks",
        lower=1,
        upper=max_num_res_blocks,
        default_value=3
    )
    res_block_type = ConfigSpace.CategoricalHyperparameter(
        'block_type',
        block_types
    )

    cs.add_hyperparameter(res_block_type)
    cs.add_hyperparameter(num_layers)
    cs.add_hyperparameter(num_res_blocks)

    cs.add_hyperparameter(
        ConfigSpace.UniformIntegerHyperparameter(
            "batch_size",
            lower=8,
            upper=256,
            default_value=16,
            log=True
        )
    )
    # Regularition parameters
    decay_type = ConfigSpace.CategoricalHyperparameter('decay_type', decay_scheduler)
    cs.add_hyperparameter(decay_type)

    mixout = ConfigSpace.CategoricalHyperparameter('mixout', include_hyperparameter)
    mixout_alpha = ConfigSpace.UniformFloatHyperparameter('mixout_alpha',
                                                          lower=0,
                                                          upper=1,
                                                          default_value=0.2
                                                          )
    cs.add_hyperparameter(mixout)
    cs.add_hyperparameter(mixout_alpha)
    cs.add_condition(ConfigSpace.EqualsCondition(mixout_alpha, mixout, 'Yes'))

    shake_shake = ConfigSpace.CategoricalHyperparameter('shake-shake', include_hyperparameter)
    cs.add_hyperparameter(shake_shake)

    cs.add_hyperparameter(
        ConfigSpace.UniformFloatHyperparameter(
            "learning_rate",
            lower=10e-4,
            upper=10e-1,
            default_value=10e-2,
            log=True
        )
    )

    optimizer = ConfigSpace.CategoricalHyperparameter(
        'optimizer',
        optimizers
    )
    momentum = ConfigSpace.UniformFloatHyperparameter(
        "momentum",
        lower=0.0,
        upper=0.9,
        default_value=0.9
    )
    cs.add_hyperparameter(optimizer)
    cs.add_hyperparameter(momentum)

    cs.add_condition(
        ConfigSpace.OrConjunction(
            ConfigSpace.EqualsCondition(
                momentum,
                optimizer,
                'SGDW'
            ),
            ConfigSpace.EqualsCondition(
                momentum,
                optimizer,
                'SGD'
            )
        )
    )
    weight_decay = ConfigSpace.UniformFloatHyperparameter(
        "weight_decay",
        lower=10e-5,
        upper=10e-3,
        default_value=10e-4
    )
    activate_weight_decay = ConfigSpace.CategoricalHyperparameter(
        'activate_weight_decay',
        include_hyperparameter
    )
    cs.add_hyperparameter(weight_decay)
    cs.add_hyperparameter(activate_weight_decay)
    cs.add_condition(
        ConfigSpace.EqualsCondition(
            weight_decay,
            activate_weight_decay,
            'Yes'
        )
    )

    # it is the upper bound of the nr of layers,
    # since the configuration will actually be sampled.
    for i in range(1, max_num_layers + 1):

        n_units = ConfigSpace.UniformIntegerHyperparameter(
            "num_units_%d" % i,
            lower=16,
            upper=256,
            default_value=64,
            log=True
        )
        cs.add_hyperparameter(n_units)
        cs.add_condition(
            ConfigSpace.OrConjunction(
                ConfigSpace.GreaterThanCondition(
                    n_units,
                    num_layers,
                    i
                ),
                ConfigSpace.EqualsCondition(
                    n_units,
                    num_layers,
                    i
                )
            )
        )

    # add drop out for the number of residual blocks
    for i in range(1, max_num_res_blocks + 1):

        dropout = ConfigSpace.UniformFloatHyperparameter(
            "dropout_%d" % i,
            lower=0,
            upper=0.9,
            default_value=0.5
        )
        cs.add_hyperparameter(dropout)
        cs.add_condition(
            ConfigSpace.OrConjunction(
                ConfigSpace.GreaterThanCondition(
                    dropout,
                    num_res_blocks,
                    i
                ),
                ConfigSpace.EqualsCondition(
                    dropout,
                    num_res_blocks,
                    i
                )
            )
        )

    return cs


def validate_output(x):

    return x != x


def train(config, num_epochs, x_train, y_train, x_val, y_val, x_test, y_test):

    logger = logging.getLogger(__name__)

    # number of dataset classes
    nr_classes = max(y_train) + 1

    # Get the batch size
    batch_size = config["batch_size"]

    device = torch.device("cuda"
                          if torch.cuda.is_available() else "cpu")

    network = FcResNet(config, x_train.shape[1],
                       nr_classes).to(device)

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
            weight_decay=weight_decay
        )
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=weight_decay
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
        config['activate_weight_decay'],
        config['decay_type']
    )
    logger.info('FcResNet started training')

    # array to save the validation loss for each epoch
    network_val_loss = []
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
            if config['mixout'] == 'Yes':
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
            loss_function = utilities.regularization.mixup_criterion(targets_a, targets_b, lam)
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

        outputs = network(x_val)
        val_loss = criterion(outputs, y_val).item()
        network_train_loss.append(running_loss / nr_batches)
        network_val_loss.append(val_loss)
        logger.info('Epoch %d, Train loss: %.3f, Validation loss: %.3f',
                    epoch + 1, running_loss / nr_batches, val_loss)
        logger.info('Learning rate: %.3f',
                    scheduled_optimizer.get_learning_rate())
        logger.info('Weight decay: %.3f',
                    scheduled_optimizer.get_weight_decay())

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
        'validation': network_val_loss,
        'train': network_train_loss
    }
    return output_information


class FcResNet(nn.Module):

    def __init__(self, config, input_features, nr_labels, number_epochs=100):

        super(FcResNet, self).__init__()
        self.config = config
        self.number_epochs = number_epochs
        # create the residual blocks
        self.layers = self._make_layer(self.config["num_res_blocks"], input_features)
        self.fc_layer = nn.Linear(self.config["num_units_%i" % self.config["num_layers"]], int(nr_labels))
        self.softmax_layer = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out = self.layers(x)
        out = self.fc_layer(out)
        out = self.softmax_layer(out)
        return out

    def _make_layer(self, num_res_blocks, input_features):

        layer = list()
        layer.append(BasicBlock(input_features, self.config, 1))
        for i in range(2, num_res_blocks + 1):
            layer.append(BasicBlock(self.config["num_units_%i" % self.config["num_layers"]], self.config, i))
        return nn.Sequential(*layer)


class PreActResPath(nn.Module):

    def __init__(self, in_features, config, block_nr):

        super(PreActResPath, self).__init__()
        self.number_layers = config["num_layers"]
        self.relu = nn.ReLU(inplace=True)
        self.projection = None
        setattr(self, "b_norm_1", nn.BatchNorm1d(in_features))
        setattr(self, "fc_1", nn.Linear(in_features, config["num_units_1"]))

        # if 'dropout_1' in config:
        # setattr(self, 'dropout_1', nn.Dropout(p=config['dropout_1']))

        # adding dropout only in the case of a 2 layer res block and only once
        # TODO generalize
        setattr(self, 'dropout_1', nn.Dropout(p=config['dropout_%d' % block_nr]))

        for i in range(2, self.number_layers + 1):
            setattr(self, "b_norm_%d" % i, nn.BatchNorm1d(config["num_units_%d" % (i - 1)]))
            setattr(self, "fc_%d" % i, nn.Linear(config["num_units_%d" % (i - 1)], config["num_units_%d" % i]))
            # if 'dropout_%d' % i in config:
            # setattr(self, 'dropout_%d' % i, nn.Dropout(p=config['dropout_%d' % i]))

        if in_features != config["num_units_%d" % self.number_layers]:
            self.projection = nn.Linear(in_features, config["num_units_%d" % self.number_layers])

    def forward(self, x):

        out = x

        for i in range(1, self.number_layers + 1):
            out = getattr(self, 'b_norm_%d' % i)(out)
            out = self.relu(out)
            out = getattr(self, 'fc_%d' % i)(out)
            if getattr(self, 'dropout_%d' % i, None) is not None:
                out = getattr(self, 'dropout_%d' % i)(out)

        return out


class BasicResPath(nn.Module):

    def __init__(self, in_features, config, block_nr):

        super(BasicResPath, self).__init__()
        self.number_layers = config["num_layers"]
        self.relu = nn.ReLU(inplace=True)
        self.projection = None
        setattr(self, "fc_1", nn.Linear(in_features, config["num_units_1"]))
        setattr(self, "b_norm_1", nn.BatchNorm1d(config["num_units_1"]))

        # adding dropout only in the case of a 2 layer res block and only once
        # TODO generalize
        setattr(self, 'dropout_1', nn.Dropout(p=config['dropout_%d' % block_nr]))

        for i in range(2, self.number_layers + 1):
            setattr(self, 'fc_%d' % i, nn.Linear(config["num_units_%d" % (i - 1)], config["num_units_%d" % i]))
            setattr(self, 'b_norm_%d' % i, nn.BatchNorm1d(config["num_units_%d" % i]))
            # if 'dropout_%d' % i in config:
            # setattr(self, 'dropout_%d' % i, nn.Dropout(p=config['dropout_%d' % i]))

    def forward(self, x):

        out = x
        for i in range(1, self.number_layers):
            out = getattr(self, 'fc_%d' % i)(out)
            out = getattr(self, 'b_norm_%d' % i)(out)
            out = self.relu(out)
            if getattr(self, 'dropout_%d' % i, None) is not None:
                out = getattr(self, 'dropout_%d' % i)(out)

        out = getattr(self, 'fc_%d' % self.number_layers)(out)
        out = getattr(self, 'b_norm_%d' % self.number_layers)(out)

        return out


class BasicBlock(nn.Module):

    def __init__(self, in_features, config, block_nr):

        super(BasicBlock, self).__init__()

        self.training = True
        self.relu = nn.ReLU(inplace=True)
        self.number_layers = config["num_layers"]

        # TODO configuration should be taken not hardcoded
        self.shake_config = (True, True, True)
        self.block_type = config['block_type']

        if self.block_type == 'BasicRes':
            res_path = BasicResPath
        elif self.block_type == 'PreRes':
            res_path = PreActResPath
        else:
            raise ValueError("Unexpected residual block type")

        if config['shake-shake'] == 'Yes':

            self.shake_shake = True
            self.residual_path1 = res_path(in_features, config, block_nr)
            self.residual_path2 = res_path(in_features, config, block_nr)

        else:
            self.residual_path1 = res_path(in_features, config, block_nr)
            self.shake_shake = False

        if in_features != config["num_units_%d" % self.number_layers]:
            self.projection = nn.Linear(in_features, config["num_units_%d" % self.number_layers])
        else:
            self.projection = None

    def forward(self, x):

        residual = x

        if self.shake_shake:

            x1 = self.residual_path1(x)
            x2 = self.residual_path2(x)

            if self.training:

                shake_config = self.shake_config

            else:

                shake_config = (False, False, False)

            alpha, beta = utilities.regularization.get_alpha_beta(x.size(0), shake_config, x.is_cuda)
            out = utilities.regularization.shake_function(x1, x2, alpha, beta)

        else:

            out = self.residual_path1(x)

        if self.projection is not None:
            residual = self.projection(residual)

        out += residual

        if self.block_type == 'BasicRes':
            out = self.relu(out)

        return out
