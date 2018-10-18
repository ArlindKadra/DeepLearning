import torch.nn as nn


class FcNet(nn.Module):

    def __init__(self, config, input_features, nr_labels):

        super(FcNet, self).__init__()
        self.config = config
        # create the blocks
        self.layers = self._make_block(self.config["num_layers"], input_features)
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

    def _make_block(self, nr_layers, input_features):

        blocks = list()
        blocks.append(BasicBlock(input_features, self.config, 1))
        for i in range(2, nr_layers + 1):
            blocks.append(BasicBlock(self.config["num_units_%i" % (i-1)], self.config, i))
        return nn.Sequential(*blocks)


class BasicBlock(nn.Module):

    def __init__(self, in_features, config, block_nr):

        super(BasicBlock, self).__init__()
        self.dropout = True if config['activate_dropout'] == 'Yes' else False
        self.training = True
        self.linear = nn.Linear(in_features, config['num_units_%i' % block_nr])
        self.relu = nn.ReLU(inplace=True)
        if self.dropout:
            self.dropout = nn.Dropout(p=config['dropout_%i' % block_nr])
        self.batch_norm = nn.BatchNorm1d(config['num_units_%i' % block_nr])

    def forward(self, x):

        out = self.linear(x)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.batch_norm(out)

        return out
