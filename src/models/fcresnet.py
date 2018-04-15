import ConfigSpace
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import src.utils as utils


def get_config_space(max_num_layers=3, max_num_res_blocks=30):

    cs = ConfigSpace.ConfigurationSpace()

    num_layers = ConfigSpace.UniformIntegerHyperparameter("num_layers",
                                                          lower=1,
                                                          upper=max_num_layers,
                                                          default_value=2)
    cs.add_hyperparameter(num_layers)
    num_res_blocks = ConfigSpace.UniformIntegerHyperparameter("num_res_blocks",
                                                              lower=10,
                                                              upper=max_num_res_blocks,
                                                              default_value=10)
    cs.add_hyperparameter(num_res_blocks)
    cs.add_hyperparameter(ConfigSpace.UniformIntegerHyperparameter("batch_size",
                                                                   lower=8,
                                                                   upper=256,
                                                                   default_value=16,
                                                                   log=True))
    cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter("learning_rate",
                                                                 lower=10e-6,
                                                                 upper=10e-1,
                                                                 default_value=10e-2,
                                                                 log=True))
    cs.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter("momentum",
                                                                 lower=0.0,
                                                                 upper=0.9,
                                                                 default_value=0.9))

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

        l2_reg = ConfigSpace.UniformFloatHyperparameter("l2_reg_%d" % i,
                                                        lower=10e-6,
                                                        upper=10e-2,
                                                        default_value=10e-4)
        cs.add_hyperparameter(l2_reg)

        if i > 1:
            cond = ConfigSpace.GreaterThanCondition(n_units, num_layers, i - 1)
            cs.add_condition(cond)

            cond = ConfigSpace.GreaterThanCondition(dropout, num_layers, i - 1)
            cs.add_condition(cond)

            cond = ConfigSpace.GreaterThanCondition(l2_reg, num_layers, i - 1)
            cs.add_condition(cond)

    return cs


def validate_output(x):

    return x != x

def train(config, num_epochs, x_train, y_train, x_test, y_test):

    nr_classes = max(y_train) + 1
    batch_size = config["batch_size"]
    network = FcResNet(BasicBlock, config, x_train.shape[1], nr_classes)
    # network.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), config["learning_rate"], momentum=config["momentum"])

    # loop over the dataset according to the number of epochs
    for epoch in range(0, num_epochs):

        running_loss = 0.0
        nr_batches = 0
        for i in range(0, (x_train.shape[0] - batch_size), batch_size):
            # get the inputs
            x = x_train[i:i + batch_size]
            y = y_train[i:i + batch_size]
            x = torch.from_numpy(x)
            y = torch.from_numpy(y).long()
            # wrap them in Variable
            x, y = Variable(x), Variable(y)
            # x = x.cuda()
            # y = y.cuda()
            # forward + backward + optimize
            optimizer.zero_grad()  # zero the gradient buffers
            output = network(x)

            # stop training if we have NaN values in the output
            if utils.contains_nan(output.data.numpy()):
                raise ValueError("NaN value in output")

            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            nr_batches += 1
        print('Epoch %d, loss: %.3f' % (epoch + 1, running_loss / nr_batches))

    correct = 0
    total = 0
    x_test = Variable(torch.from_numpy(x_test))
    # x_test.cuda()
    outputs = network(x_test)
    y_test = torch.from_numpy(y_test).long()
    # y_test.cuda()
    _, predicted = torch.max(outputs.data, 1)
    total += y_test.size(0)
    correct += (predicted == y_test).sum()
    accuracy = 100 * correct / total
    print('Accuracy of the network: %d %%' % accuracy)
    print('Finished Training')
    return accuracy


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
        self.fc_layers = []
        self.batch_norm_layers = []
        self.relu = nn.ReLU(inplace=True)
        self.fc_layers.append(nn.Linear(in_features, config["num_units_1"]))
        self.batch_norm_layers.append(nn.BatchNorm1d(config["num_units_1"]))

        for i in range(2, config["num_layers"] + 1):
            self.fc_layers.append(nn.Linear(config["num_units_%d" % (i-1)], config["num_units_%d" % i]))
            self.batch_norm_layers.append(nn.BatchNorm1d(config["num_units_%d" % i]))

    def forward(self, x):

        residual = x
        out = x
        for i in range(0, len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.batch_norm_layers[i](out)
            out = self.relu(out)
        out = self.fc_layers[len(self.fc_layers) - 1](out)
        out = self.batch_norm_layers[len(self.fc_layers) - 1](out)
        if residual.size()[1] != out.size()[1]:
            projection = nn.Linear(residual.size()[1], out.size()[1])
            residual = projection(residual)
        out += residual
        out = self.relu(out)
        return out
