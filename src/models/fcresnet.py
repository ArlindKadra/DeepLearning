import utils
from optim.adamw import AdamW
from optim.sgdw import SGDW
from optim.lr_scheduler import ScheduledOptimizer, CosineScheduler
import logging
import ConfigSpace
import torch
import torch.nn as nn

# TODO Max number of layers and res blocks should be read from the config file
def get_config_space(max_num_layers=2, max_num_res_blocks=15):

    optimizers = ['SGD', 'AdamW']
    dropout_values = ['Yes', 'No']
    block_types = ['BasicRes', 'PreRes']

    cs = ConfigSpace.ConfigurationSpace()

    num_layers = ConfigSpace.UniformIntegerHyperparameter("num_layers",
                                                          lower=1,
                                                          upper=max_num_layers,
                                                          default_value=2)
    cs.add_hyperparameter(num_layers)
    num_res_blocks = ConfigSpace.UniformIntegerHyperparameter("num_res_blocks",
                                                              lower=5,
                                                              upper=max_num_res_blocks,
                                                              default_value=5)
    cs.add_hyperparameter(num_res_blocks)
    cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("batch_size",
                                                                   lower=8,
                                                                   upper=256,
                                                                   default_value=16,
                                                                   log=True))
    res_block_type = ConfigSpace.CategoricalHyperparameter('block_type', block_types)
    cs.add_hyperparameter(res_block_type)
    cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter("learning_rate",
                                                                 lower=10e-4,
                                                                 upper=10e-1,
                                                                 default_value=10e-2,
                                                                 log=True))
    optimizer = ConfigSpace.CategoricalHyperparameter('optimizer', optimizers)
    cs.add_hyperparameter(optimizer)

    momentum = ConfigSpace.UniformFloatHyperparameter("momentum",
                                                      lower=0.0,
                                                      upper=0.9,
                                                      default_value=0.9)
    cs.add_hyperparameter(momentum)
    cs.add_condition(ConfigSpace.EqualsCondition(momentum, optimizer, 'SGD'))

    l2_reg = ConfigSpace.UniformFloatHyperparameter("l2_reg",
                                                    lower=10e-6,
                                                    upper=10e-2,
                                                    default_value=10e-4)
    cs.add_hyperparameter(l2_reg)

    weight_decay = ConfigSpace.UniformFloatHyperparameter("weight_decay",
                                                          lower=10e-5,
                                                          upper=10e-3,
                                                          default_value=10e-4)
    cs.add_hyperparameter(weight_decay)
    dropout_flag = ConfigSpace.CategoricalHyperparameter('dropout', dropout_values)
    cs.add_hyperparameter(dropout_flag)

    # it is the upper bound of the nr of layers, since the configuration will actually be sampled.
    for i in range(1, max_num_layers + 1):

        n_units = ConfigSpace.UniformIntegerHyperparameter("num_units_%d" % i,
                                                           lower=128,
                                                           upper=1024,
                                                           default_value=128,
                                                           log=True)
        cs.add_hyperparameter(n_units)

        dropout = ConfigSpace.UniformFloatHyperparameter("dropout_%d" % i,
                                                         lower=0.0,
                                                         upper=0.9,
                                                         default_value=0.5)
        cs.add_hyperparameter(dropout)
        dropout_cond = ConfigSpace.EqualsCondition(dropout, dropout_flag, 'Yes')

        if i >= 1:
            cond = ConfigSpace.GreaterThanCondition(n_units, num_layers, i)
            equals_cond = ConfigSpace.EqualsCondition(n_units, num_layers, i)
            cs.add_condition(ConfigSpace.OrConjunction(cond, equals_cond))
            # every 2 fully connected layers / 1 dropout layer in between
            cond = ConfigSpace.GreaterThanCondition(dropout, num_layers, i)
            cs.add_condition(ConfigSpace.AndConjunction(cond, dropout_cond))

    return cs


def validate_output(x):

    return x != x


def train(config, num_epochs, x_train, y_train, x_val, y_val, x_test, y_test):

    logger = logging.getLogger(__name__)
    nr_classes = max(y_train) + 1
    batch_size = config["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config['block_type'] == 'BasicRes':
        network = FcResNet(BasicBlock, config, x_train.shape[1], nr_classes).to(device)
    elif config['block_type'] == 'PreRes':
        network = FcResNet(PreActBlock, config, x_train.shape[1], nr_classes).to(device)
    else:
        logger.error("Unexpected residual block type")
        raise ValueError("Unexpected residual block type")

    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    logger.info("Number of network parameters %d", total_params)
    loss_function = nn.CrossEntropyLoss()

    if config['optimizer'] == 'SGD':
        optimizer = SGDW(network.parameters(), lr=config["learning_rate"],
                         momentum=config["momentum"], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = AdamW(network.parameters(), lr=config['learning_rate'],
                          l2_decay=config['l2_reg'], weight_decay=config['weight_decay'])
    else:
        logger.error("Unexpected optimizer value")
        raise ValueError("Unexpected optimizer value")

    anneal_max_epoch = int(1 / 10 * num_epochs)
    anneal_multiply = 2
    anneal_epoch = 0
    scheduled_optimizer = ScheduledOptimizer(optimizer, CosineScheduler)
    logger.info('FcResNet started training')
    # Save the validation accuracy for each epoch
    network_val_loss = []
    x_val = torch.from_numpy(x_val)
    x_val.requires_grad_(False)
    y_val = torch.from_numpy(y_val)
    y_val.requires_grad_(False)
    x_val, y_val = x_val.to(device), y_val.to(device)
    # loop over the dataset according to the number of epochs
    for epoch in range(0, num_epochs):

        running_loss = 0.0
        nr_batches = 0
        # train the network
        for i in range(0, (x_train.shape[0] - batch_size), batch_size):
            # get the inputs
            x = x_train[i:i + batch_size]
            y = y_train[i:i + batch_size]
            x = torch.from_numpy(x)
            y = torch.from_numpy(y).long()
            x, y = x.to(device), y.to(device)
            # forward + backward + optimize
            scheduled_optimizer.zero_grad()  # zero the gradient buffers
            output = network(x)

            # stop training if we have NaN values in the output
            if utils.contains_nan((output.cpu()).data.numpy()):
                # TODO switch to logger exception
                logger.error('Output contains NaN values')
                raise ValueError("NaN value in output")

            loss = loss_function(output, y)
            loss.backward()
            scheduled_optimizer.step(anneal_epoch / anneal_max_epoch)
            running_loss += loss.item()
            nr_batches += 1

        outputs = network(x_val)
        val_loss = loss_function(outputs, y_val).item()
        network_val_loss.append(val_loss)
        logger.info('Epoch %d, Train loss: %.3f, Validation loss: %.3f', epoch + 1, running_loss / nr_batches, val_loss)
        logger.info('Learning rate: %.3f', scheduled_optimizer.get_learning_rate())
        logger.info('Weight decay: %.3f', scheduled_optimizer.get_weight_decay())
        # set the anneal schedule progress
        if anneal_epoch < anneal_max_epoch:
            anneal_epoch += 1
        elif anneal_epoch == anneal_max_epoch:
            anneal_epoch = 0
            anneal_max_epoch = anneal_max_epoch * anneal_multiply
        else:
            logger.info("Something went wrong, epochs > max epochs in restart")

    with torch.no_grad():
        correct = 0
        total = 0
        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(y_test).long()
        x_test, y_test = x_test.to(device), y_test.to(device)
        outputs = network(x_test)
        test_loss = loss_function(outputs, y_test)
        _, predicted = torch.max(outputs.data, 1)
        total += y_test.size(0)
        correct += ((predicted == y_test).sum()).item()
        accuracy = 100 * correct / total
    logger.info('Test loss: %.3f, accuracy of the network: %.3f %%', test_loss.item(), accuracy)
    output_information = {'test': (test_loss.item(), accuracy), 'validation': network_val_loss}
    return output_information


class FcResNet(nn.Module):

    def __init__(self, block, config, input_features, nr_labels, number_epochs=100):

        super(FcResNet, self).__init__()
        self.config = config
        self.number_epochs = number_epochs
        # create the residual blocks
        self.layers = self._make_layer(block, self.config["num_res_blocks"], input_features)
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

    def _make_layer(self, block, num_res_blocks, input_features):

        layer = list()
        layer.append(block(input_features, self.config))
        for i in range(1, num_res_blocks):
            layer.append(block(self.config["num_units_%i" % self.config["num_layers"]], self.config))
        return nn.Sequential(*layer)


class BasicBlock(nn.Module):

    def __init__(self, in_features, config):

        super(BasicBlock, self).__init__()
        self.number_layers = config["num_layers"]
        self.relu = nn.ReLU(inplace=True)
        self.projection = None
        setattr(self, "fc_1", nn.Linear(in_features, config["num_units_1"]))
        setattr(self, "b_norm_1", nn.BatchNorm1d(config["num_units_1"]))

        if 'dropout_1' in config:
            setattr(self, 'dropout_1', nn.Dropout(p=config['dropout_1']))

        for i in range(2, self.number_layers + 1):
            setattr(self, 'fc_%d' % i, nn.Linear(config["num_units_%d" % (i-1)], config["num_units_%d" % i]))
            setattr(self, 'b_norm_%d' % i, nn.BatchNorm1d(config["num_units_%d" % i]))
            if 'dropout_%d' % i in config:
                setattr(self, 'dropout_%d' % i, nn.Dropout(p=config['dropout_%d' % i]))

        if in_features != config["num_units_%d" % self.number_layers]:
            self.projection = nn.Linear(in_features, config["num_units_%d" % self.number_layers])

    def forward(self, x):

        residual = x
        out = x
        for i in range(1, self.number_layers):
            out = getattr(self, 'fc_%d' % i)(out)
            out = getattr(self, 'b_norm_%d' % i)(out)
            out = self.relu(out)
            if getattr(self, 'dropout_%d' % i, None) is not None:
                out = getattr(self, 'dropout_%d' % i)(out)

        out = getattr(self, 'fc_%d' % self.number_layers)(out)
        out = getattr(self, 'b_norm_%d' % self.number_layers)(out)
        if self.projection is not None:
            residual = self.projection(residual)

        out += residual
        out = self.relu(out)
        return out


class PreActBlock(nn.Module):

    def __init__(self, in_features, config):

        super(PreActBlock, self).__init__()

        self.number_layers = config["num_layers"]
        self.relu = nn.ReLU(inplace=True)
        self.projection = None
        setattr(self, "b_norm_1", nn.BatchNorm1d(in_features))
        setattr(self, "fc_1", nn.Linear(in_features, config["num_units_1"]))

        if 'dropout_1' in config:
            setattr(self, 'dropout_1', nn.Dropout(p=config['dropout_1']))

        for i in range(2, self.number_layers + 1):
            setattr(self, "b_norm_%d" % i, nn.BatchNorm1d(config["num_units_%d" % (i - 1)]))
            setattr(self, "fc_%d" % i, nn.Linear(config["num_units_%d" % (i - 1)], config["num_units_%d" % i]))
            if 'dropout_%d' % i in config:
                setattr(self, 'dropout_%d' % i, nn.Dropout(p=config['dropout_%d' % i]))

        if in_features != config["num_units_%d" % self.number_layers]:
            self.projection = nn.Linear(in_features, config["num_units_%d" % self.number_layers])

    def forward(self, x):

        residual = x
        out = x
        for i in range(1, self.number_layers + 1):
            out = getattr(self, 'b_norm_%d' % i)(out)
            out = self.relu(out)
            out = getattr(self, 'fc_%d' % i)(out)
            if getattr(self, 'dropout_%d' % i, None) is not None:
                out = out = getattr(self, 'dropout_%d' % i)(out)

        if self.projection is not None:
            residual = self.projection(residual)

        out += residual
        return out
