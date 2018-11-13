import ConfigSpace


def get_super_fcresnet_config(layers_block=2, num_res_blocks=18, super_blocks=3):

    # Config
    optimizers = ['SGDW', 'AdamW', 'SGD', 'Adam']
    decay_scheduler = [
        'cosine_annealing',
        'cosine_decay',
        'exponential_decay'
    ]
    include_hyperparameter = ['Yes', 'No']

    cs = ConfigSpace.ConfigurationSpace()

    # Architecture parameters
    nr_layers_block = ConfigSpace.Constant(
        "num_layers",
        layers_block
    )
    nr_res_blocks = ConfigSpace.Constant(
        "num_res_blocks",
        num_res_blocks
    )
    res_block_type = ConfigSpace.Constant(
        'block_type',
        'PreRes'
    )
    nr_super_blocks = ConfigSpace.Constant(
        'num_super_blocks',
        super_blocks

    )
    cs.add_hyperparameter(res_block_type)
    cs.add_hyperparameter(nr_layers_block)
    cs.add_hyperparameter(nr_res_blocks)
    cs.add_hyperparameter(nr_super_blocks)

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
    for i in range(1, super_blocks + 1):
        for j in range(1, layers_block + 1):
            n_units = ConfigSpace.UniformIntegerHyperparameter(
                "num_units_%d_%d" % (i, j),
                lower=16,
                upper=256,
                default_value=64,
                log=True
            )
            cs.add_hyperparameter(n_units)

    activate_dropout = cs.add_hyperparameter(
        ConfigSpace.CategoricalHyperparameter(
            'activate_dropout',
            include_hyperparameter
        )
    )
    # get dropout value for super_block
    # the same value will be used for
    # residual blocks in the super_blocks
    for i in range(1, super_blocks + 1):
        # for now only dropout between layers
        # 1 layer dropout for res block
        dropout = ConfigSpace.UniformFloatHyperparameter(
            "dropout_%d_1" % i,
            lower=0,
            upper=0.7,
            default_value=0.5
        )
        cs.add_hyperparameter(dropout)
        cs.add_condition(
            ConfigSpace.EqualsCondition(
                dropout,
                activate_dropout,
                'Yes'
            ),
        )

    return cs


def get_fixed_fcresnet_config(
        layers_block=2, num_res_blocks=17, super_blocks=1,
        activate_mixout='No', activate_shake_shake='No',
        activate_weight_decay='No', activate_dropout='No'):

    # Config
    optimizers = ['SGDW', 'AdamW', 'SGD', 'Adam']
    decay_scheduler = [
        'cosine_annealing',
        'cosine_decay',
        'exponential_decay'
    ]

    cs = ConfigSpace.ConfigurationSpace()

    # Architecture parameters
    nr_layers_block = ConfigSpace.Constant(
        "num_layers",
        layers_block
    )
    nr_res_blocks = ConfigSpace.Constant(
        "num_res_blocks",
        num_res_blocks
    )
    res_block_type = ConfigSpace.Constant(
        'block_type',
        'PreRes'
    )
    nr_super_blocks = ConfigSpace.Constant(
        'num_super_blocks',
        super_blocks

    )
    cs.add_hyperparameter(res_block_type)
    cs.add_hyperparameter(nr_layers_block)
    cs.add_hyperparameter(nr_res_blocks)
    cs.add_hyperparameter(nr_super_blocks)

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
    decay_type = ConfigSpace.CategoricalHyperparameter(
        'decay_type',
        decay_scheduler
    )
    cs.add_hyperparameter(decay_type)

    mixout = ConfigSpace.Constant(
        'mixout',
        activate_mixout
    )

    mixout_alpha = ConfigSpace.UniformFloatHyperparameter(
        'mixout_alpha',
        lower=0,
        upper=1,
        default_value=0.2
    )
    cs.add_hyperparameter(mixout)
    cs.add_hyperparameter(mixout_alpha)
    cs.add_condition(
        ConfigSpace.EqualsCondition(
            mixout_alpha,
            mixout,
            'Yes'
        )
    )

    shake_shake = ConfigSpace.Constant(
        'shake-shake',
        activate_shake_shake
    )
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
    activate_weight_decay = ConfigSpace.Constant(
        'activate_weight_decay',
        activate_weight_decay

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
    for i in range(1, super_blocks + 1):
        for j in range(1, layers_block + 1):
            n_units = ConfigSpace.Constant(
                "num_units_%d_%d" % (i, j),
                64
            )
            cs.add_hyperparameter(n_units)

    activate_dropout = cs.add_hyperparameter(
        ConfigSpace.Constant(
            'activate_dropout',
            activate_dropout
        )
    )
    # get dropout value for super_block
    # the same value will be used for
    # residual blocks in the super_blocks
    for i in range(1, super_blocks + 1):
        # for now only dropout between layers
        # 1 layer dropout for res block
        dropout = ConfigSpace.UniformFloatHyperparameter(
            "dropout_%d_1" % i,
            lower=0,
            upper=0.7,
            default_value=0.5
        )
        cs.add_hyperparameter(dropout)
        cs.add_condition(
            ConfigSpace.EqualsCondition(
                dropout,
                activate_dropout,
                'Yes'
            ),
        )

    return cs


def get_fc_config(max_nr_layers=28):

    # Config
    optimizers = ['SGDW', 'AdamW', 'SGD', 'Adam']
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
        upper=max_nr_layers,
        default_value=6)

    cs.add_hyperparameter(num_layers)

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

    activate_dropout = ConfigSpace.CategoricalHyperparameter('activate_dropout', include_hyperparameter)
    cs.add_hyperparameter(activate_dropout)
    # it is the upper bound of the nr of layers,
    # since the configuration will actually be sampled.
    for i in range(1, max_nr_layers + 1):

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

        dropout = ConfigSpace.UniformFloatHyperparameter(
            "dropout_%d" % i,
            lower=0,
            upper=0.9,
            default_value=0.5
        )
        cs.add_hyperparameter(dropout)
        dropout_cond_1 = ConfigSpace.OrConjunction(
            ConfigSpace.GreaterThanCondition(
                dropout,
                num_layers,
                i
            ),
            ConfigSpace.EqualsCondition(
                dropout,
                num_layers,
                i
            )
        )
        dropout_cond_2 = ConfigSpace.EqualsCondition(dropout, activate_dropout, 'Yes')
        cs.add_condition(
            ConfigSpace.AndConjunction(
                dropout_cond_1,
                dropout_cond_2
            )
        )

    return cs


def get_fixed_fc_config(
        max_nr_layers=34,
        activate_dropout='No',
        activate_weight_decay='No'
):

    # Config
    optimizers = [
        'Adam',
        'AdamW',
        'SGD',
        'SGDW'
    ]

    decay_scheduler = [
        'cosine_annealing',
        'cosine_decay',
        'exponential_decay'
    ]

    cs = ConfigSpace.ConfigurationSpace()

    # Architecture parameters
    num_layers = ConfigSpace.Constant(
        "num_layers",
        max_nr_layers
    )

    cs.add_hyperparameter(num_layers)

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

    cs.add_hyperparameter(
        ConfigSpace.UniformFloatHyperparameter(
            "learning_rate",
            lower=10e-4,
            upper=10e-1,
            default_value=10e-2,
            log=True
        )
    )

    lr_fraction = ConfigSpace.UniformFloatHyperparameter(
        "final_lr_fraction",
        lower=1e-4,
        upper=1.,
        default_value=1e-2,
        log=True
    )

    cs.add_hyperparameter(lr_fraction)
    cs.add_condition(
        ConfigSpace.EqualsCondition(
            lr_fraction,
            decay_type,
            'exponential_decay'

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
    weight_decay_activate = ConfigSpace.Constant(
        'activate_weight_decay',
        activate_weight_decay
    )

    if activate_weight_decay == 'Yes':
        cs.add_hyperparameter(weight_decay)
    cs.add_hyperparameter(weight_decay_activate)

    dropout = ConfigSpace.Constant(
        'activate_dropout',
        activate_dropout
    )
    cs.add_hyperparameter(dropout)
    # it is the upper bound of the nr of layers,
    # since the configuration will actually be sampled.
    for i in range(1, max_nr_layers + 1):

        n_units = ConfigSpace.Constant(
            "num_units_%d" % i,
            64
        )
        cs.add_hyperparameter(n_units)

        dropout = ConfigSpace.UniformFloatHyperparameter(
            "dropout_%d" % i,
            lower=0,
            upper=0.7,
            default_value=0.5
        )

        if activate_dropout == 'Yes':
            cs.add_hyperparameter(dropout)


    return cs
