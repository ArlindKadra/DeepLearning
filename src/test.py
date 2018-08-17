import ConfigSpace

number_layers = 2

cs = ConfigSpace.ConfigurationSpace()
dropout_values = ['True', 'False']
dropout_flag = ConfigSpace.CategoricalHyperparameter('dropout', dropout_values)
cs.add_hyperparameter(dropout_flag)

for i in range(1, number_layers + 1):

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

    dropout_cond = ConfigSpace.EqualsCondition(dropout, dropout_flag, 'True')

    if i > 1:
        cond = ConfigSpace.GreaterThanCondition(n_units, number_layers, i - 1)
        cs.add_condition(cond)
        # every 2 fully connected layers / 1 dropout layer in between
        cond = ConfigSpace.GreaterThanCondition(dropout, number_layers, i)
        cs.add_condition(ConfigSpace.AndConjunction(cond, dropout_cond))

