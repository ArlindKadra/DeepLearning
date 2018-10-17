import torch.nn as nn

import utilities.regularization


class FcResNet(nn.Module):

    def __init__(self, config, input_features, nr_labels):

        super(FcResNet, self).__init__()
        self.config = config
        # create the residual blocks
        self.layers = self._make_layer(self.config["num_res_blocks"], input_features)
        self.fc_layer = nn.Linear(self.config["num_units_%i" % self.config["num_layers"]], int(nr_labels))
        self.softmax_layer = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
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
