import torch.nn as nn

import utilities.regularization


class FcResNet(nn.Module):

    def __init__(self, config, input_features, nr_labels):

        super(FcResNet, self).__init__()
        self.config = config
        self.activate_batch_norm = True if config['activate_batch_norm'] == 'Yes' else False
        self.relu = nn.ReLU(inplace=True)
        # create the residual blocks
        self.input_layer = nn.Linear(input_features, self.config['input_layer_units'])
        if self.activate_batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(self.config['input_layer_units'])
        self.layers = self._make_layer(self.config["num_res_blocks"], self.config['input_layer_units'])
        self.fc_layer = nn.Linear(
            self.config["num_units_%i_%i" % (
                self.config['num_super_blocks'],
                self.config["num_layers"]
            )],
            int(nr_labels)
        )
        self.softmax_layer = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out = self.input_layer(x)
        if self.activate_batch_norm:
            out = self.batch_norm_layer(out)
        out = self.relu(out)
        out = self.layers(out)
        out = self.fc_layer(out)
        out = self.softmax_layer(out)
        return out

    def _make_layer(self, num_res_blocks, input_features_res_block):

        nr_super_blocks = self.config['num_super_blocks']
        nr_res_blocks_in_superblock = int(num_res_blocks / nr_super_blocks)
        layer = list()

        layer.append(BasicBlock(input_features_res_block, self.config, 1))
        for i in range(2, nr_res_blocks_in_superblock + 1):
            layer.append(BasicBlock(self.config["num_units_%i_%i" % (1, self.config["num_layers"])], self.config, 1))

        if nr_super_blocks > 1:
            for super_block in range(2, nr_super_blocks + 1):
                layer.append(BasicBlock(self.config["num_units_%i_%i" % (super_block - 1, self.config["num_layers"])],
                                        self.config, super_block))
                for i in range(2, nr_res_blocks_in_superblock + 1):
                    layer.append(BasicBlock(self.config["num_units_%i_%i" % (super_block, self.config["num_layers"])], self.config, super_block))

        return nn.Sequential(*layer)


class PreActResPath(nn.Module):

    def __init__(self, in_features, config, super_block):

        super(PreActResPath, self).__init__()
        self.number_layers = config["num_layers"]
        self.activate_dropout = True if config['activate_dropout'] == 'Yes' else False
        self.activate_batch_norm = True if config['activate_batch_norm'] == 'Yes' else False
        self.relu = nn.ReLU(inplace=True)

        if self.activate_batch_norm:
            setattr(self, "b_norm_1", nn.BatchNorm1d(in_features))
        setattr(self, "fc_1", nn.Linear(in_features, config["num_units_%d_1" % super_block]))

        # if 'dropout_1' in config:
        # setattr(self, 'dropout_1', nn.Dropout(p=config['dropout_1']))

        # adding dropout only in the case of a 2 layer res block and only once
        # TODO generalize
        if self.activate_dropout:
            setattr(self, 'dropout_1', nn.Dropout(p=config['dropout_%d_1' % super_block]))

        for i in range(2, self.number_layers + 1):
            if self.activate_batch_norm:
                setattr(self, "b_norm_%d" % i, nn.BatchNorm1d(config["num_units_%d_%d" % (super_block, (i - 1))]))
            setattr(self, "fc_%d" % i, nn.Linear(config["num_units_%d_%d" % (super_block, (i - 1))], config["num_units_%d_%d" % (super_block, i)]))
            # if 'dropout_%d' % i in config:
            # setattr(self, 'dropout_%d' % i, nn.Dropout(p=config['dropout_%d' % i]))

    def forward(self, x):

        out = x

        for i in range(1, self.number_layers + 1):
            if self.activate_batch_norm:
                out = getattr(self, 'b_norm_%d' % i)(out)
            out = self.relu(out)
            out = getattr(self, 'fc_%d' % i)(out)
            if self.activate_dropout:
                out = getattr(self, 'dropout_%d' % i)(out)

        return out


class BasicResPath(nn.Module):

    def __init__(self, in_features, config, super_block):

        super(BasicResPath, self).__init__()
        self.number_layers = config["num_layers"]
        self.relu = nn.ReLU(inplace=True)
        self.activate_dropout = True if config['activate_dropout'] == 'Yes' else False
        self.activate_batch_norm = True if config['activate_batch_norm'] == 'Yes' else False
        setattr(self, "fc_1", nn.Linear(in_features, config["num_units_%d_1" % super_block]))

        if self.activate_batch_norm:
            setattr(self, "b_norm_1", nn.BatchNorm1d(config["num_units_%d_1" % super_block]))

        # adding dropout only in the case of a 2 layer res block and only once
        # TODO generalize
        if self.activate_dropout:
            setattr(self, 'dropout_1', nn.Dropout(p=config['dropout_%d_1' % super_block]))

        for i in range(2, self.number_layers + 1):
            setattr(self, 'fc_%d' % i, nn.Linear(config["num_units_%d_%d" % (super_block, (i - 1))], config["num_units_%d_%d" % (super_block, i)]))
            if self.activate_batch_norm:
                setattr(self, 'b_norm_%d' % i, nn.BatchNorm1d(config["num_units_%d_%d" % (super_block, i)]))
            # if 'dropout_%d' % i in config:
            # setattr(self, 'dropout_%d' % i, nn.Dropout(p=config['dropout_%d' % i]))

    def forward(self, x):

        out = x
        for i in range(1, self.number_layers):
            out = getattr(self, 'fc_%d' % i)(out)
            if self.activate_batch_norm:
                out = getattr(self, 'b_norm_%d' % i)(out)
            out = self.relu(out)
            if self.activate_dropout:
                out = getattr(self, 'dropout_%d' % i)(out)

        out = getattr(self, 'fc_%d' % self.number_layers)(out)
        if self.activate_batch_norm:
            out = getattr(self, 'b_norm_%d' % self.number_layers)(out)

        return out


class BasicBlock(nn.Module):

    def __init__(self, in_features, config, super_block_nr):

        super(BasicBlock, self).__init__()

        self.training = True
        self.relu = nn.ReLU(inplace=True)
        self.activate_batch_norm = True if config['activate_batch_norm'] == 'Yes' else False
        self.number_layers = config["num_layers"]
        self.block_type = config['block_type']

        if self.block_type == 'BasicRes':
            res_path = BasicResPath
        elif self.block_type == 'PreRes':
            res_path = PreActResPath
        else:
            raise ValueError("Unexpected residual block type")

        if config['shake-shake'] == 'Yes':

            self.shake_shake = True
            self.shake_config = config['shake_config']
            self.residual_path1 = res_path(in_features, config, super_block_nr)
            self.residual_path2 = res_path(in_features, config, super_block_nr)

        else:
            self.residual_path1 = res_path(in_features, config, super_block_nr)
            self.shake_shake = False

        if in_features != config["num_units_%d_%d" % (super_block_nr, self.number_layers)]:

            if self.activate_batch_norm:
                self.projection = nn.Sequential(
                    nn.Linear(in_features, config["num_units_%d_%d" % (super_block_nr, self.number_layers)]),
                    nn.BatchNorm1d(config["num_units_%d_%d" % (super_block_nr, self.number_layers)])
                )
            else:
                self.projection = nn.Sequential(
                    nn.Linear(in_features, config["num_units_%d_%d" % (super_block_nr, self.number_layers)])
                )
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

                shake_config = 'NNN'

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
