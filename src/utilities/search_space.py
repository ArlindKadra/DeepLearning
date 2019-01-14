import ConfigSpace


# architecture search and hyperparameter optimization
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


# hyperparameter optimization with user chosen
# methods, fixed architecture
def get_fixed_fcresnet_config(
        nr_features, feature_type,
        layers_block=2, num_res_blocks=9,
        super_blocks=1, nr_units=64,
        activate_mixout='No', activate_shake_shake='No',
        activate_weight_decay='No', activate_dropout='No',
        activate_batch_norm='No'
):

    # Config
    optimizers = ['Adam', 'AdamW', 'SGD', 'SGDW']

    shake_shake_config = [
        'YYY',
        'YNY',
        'YYN',
        'YNN'
    ]

    decay_scheduler = [
        'cosine_annealing',
        'cosine_decay',
        'exponential_decay'
    ]

    include_hyperparameter = [
        'Yes',
        'No'
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
        'BasicRes'
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

    cs.add_hyperparameter(
        ConfigSpace.CategoricalHyperparameter(
            'class_weights',
            include_hyperparameter
        )
    )

    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'feature_type',
            feature_type
        )
    )

    feature_preprocessing = ConfigSpace.CategoricalHyperparameter(
        'feature_preprocessing',
        include_hyperparameter
    )

    number_pca_components = ConfigSpace.UniformIntegerHyperparameter(
        'pca_components',
        lower=2,
        upper=nr_features,
        default_value=nr_features - 1
    )

    cs.add_hyperparameter(feature_preprocessing)
    cs.add_hyperparameter(number_pca_components)

    cs.add_condition(
        ConfigSpace.EqualsCondition(
            number_pca_components,
            feature_preprocessing,
            'Yes'
        )
    )

    # Regularition parameters
    decay_type = ConfigSpace.CategoricalHyperparameter(
        'decay_type',
        decay_scheduler
    )
    cs.add_hyperparameter(decay_type)

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

    # add constant about mixup status
    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'mixout',
            activate_mixout
        )
    )

    # add value if mixup is activated
    if activate_mixout == 'Yes':
        cs.add_hyperparameter(
            ConfigSpace.UniformFloatHyperparameter(
                'mixout_alpha',
                lower=0,
                upper=1,
                default_value=0.2
            )
        )

    # add constant related to shake shakes
    # status
    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'shake-shake',
            activate_shake_shake
        )
    )

    # add config if shake shake is activated
    if activate_shake_shake == 'Yes':
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter(
                'shake_config',
                shake_shake_config
            )
        )

    # add constant about batch norm status
    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'activate_batch_norm',
            activate_batch_norm
        )
    )

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

    # add constant about weight decay status
    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'activate_weight_decay',
            activate_weight_decay
        )
    )
    # add weight decay value if active
    if activate_weight_decay == 'Yes':
        cs.add_hyperparameter(
            ConfigSpace.UniformFloatHyperparameter(
                "weight_decay",
                lower=10e-5,
                upper=10e-3,
                default_value=10e-4
            )
        )

    # nr units for input layer
    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'input_layer_units',
            nr_units
        )
    )

    # it is the upper bound of the nr of layers,
    # since the configuration will actually be sampled.
    for i in range(1, super_blocks + 1):
        for j in range(1, layers_block + 1):
            n_units = ConfigSpace.Constant(
                "num_units_%d_%d" % (i, j),
                nr_units
            )
            cs.add_hyperparameter(n_units)

    # add constant regarding dropout
    # value
    cs.add_hyperparameter(
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

        # if dropout active
        # add value
        if activate_dropout == 'Yes':
            cs.add_hyperparameter(
                ConfigSpace.UniformFloatHyperparameter(
                    "dropout_%d_1" % i,
                    lower=0,
                    upper=0.7,
                    default_value=0.5
                )
            )

    return cs


# conditional hyperparameter optimization,
# fixed architecture
def get_fixed_conditional_fcresnet_config(
        nr_features, feature_type,
        layers_block=2, num_res_blocks=9,
        super_blocks=1, nr_units=64
):

    # Config
    optimizers = ['Adam', 'AdamW', 'SGD', 'SGDW']

    shake_shake_config = [
        'YYY',
        'YNY',
        'YYN',
        'YNN'
    ]

    decay_scheduler = [
        'cosine_annealing',
        'cosine_decay',
        'exponential_decay'
    ]

    include_hyperparameter = [
        'Yes',
        'No'
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
        'BasicRes'
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

# class weights and feature preprocessing
    cs.add_hyperparameter(
        ConfigSpace.CategoricalHyperparameter(
            'class_weights',
            include_hyperparameter
        )
    )

    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'feature_type',
            feature_type
        )
    )

    feature_preprocessing = ConfigSpace.CategoricalHyperparameter(
        'feature_preprocessing',
        include_hyperparameter
    )

    number_pca_components = ConfigSpace.UniformIntegerHyperparameter(
        'pca_components',
        lower=2,
        upper=nr_features,
        default_value=nr_features - 1
    )

    cs.add_hyperparameter(feature_preprocessing)
    cs.add_hyperparameter(number_pca_components)

    cs.add_condition(
        ConfigSpace.EqualsCondition(
            number_pca_components,
            feature_preprocessing,
            'Yes'
        )
    )

    # Regularition parameters
    decay_type = ConfigSpace.CategoricalHyperparameter(
        'decay_type',
        decay_scheduler
    )
    cs.add_hyperparameter(decay_type)

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

    # add mixup regularization status
    mixup = ConfigSpace.CategoricalHyperparameter(
        'mixout',
        include_hyperparameter
    )
    # add mixup alpha value
    mixup_alpha = ConfigSpace.UniformFloatHyperparameter(
        'mixout_alpha',
        lower=0,
        upper=1,
        default_value=0.2
    )

    cs.add_hyperparameter(mixup)
    cs.add_hyperparameter(mixup_alpha)

    # add mixup_alpha only if mixup is active
    cs.add_condition(
        ConfigSpace.EqualsCondition(
            mixup_alpha,
            mixup,
            'Yes'
        )
    )

    # shake-shake status
    shake_shake = ConfigSpace.CategoricalHyperparameter(
        'shake-shake',
        include_hyperparameter
    )
    # shake-shake config
    shake_config = ConfigSpace.CategoricalHyperparameter(
        'shake_config',
        shake_shake_config
    )

    cs.add_hyperparameter(shake_shake)
    cs.add_hyperparameter(shake_config)

    # add shake-shake config only if shake-shake is active
    cs.add_condition(
        ConfigSpace.EqualsCondition(
            shake_config,
            shake_shake,
            'Yes'
        )
    )

    # batch norm status
    cs.add_hyperparameter(
        ConfigSpace.CategoricalHyperparameter(
            'activate_batch_norm',
            include_hyperparameter
        )
    )

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

    # weight decay status
    weight_decay = ConfigSpace.CategoricalHyperparameter(
            'activate_weight_decay',
            include_hyperparameter
    )
    # weight decay value
    weight_decay_value = ConfigSpace.UniformFloatHyperparameter(
        "weight_decay",
        lower=10e-5,
        upper=10e-3,
        default_value=10e-4
    )

    cs.add_hyperparameter(weight_decay)
    cs.add_hyperparameter(weight_decay_value)

    # add weight decay value only if active
    cs.add_condition(
        ConfigSpace.EqualsCondition(
            weight_decay_value,
            weight_decay,
            'Yes'
        )
    )

    # nr units for input layer
    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'input_layer_units',
            nr_units
        )
    )

    # it is the upper bound of the nr of layers,
    # since the configuration will actually be sampled.
    for i in range(1, super_blocks + 1):
        for j in range(1, layers_block + 1):
            n_units = ConfigSpace.Constant(
                "num_units_%d_%d" % (i, j),
                nr_units
            )
            cs.add_hyperparameter(n_units)

    # dropout status
    dropout = ConfigSpace.CategoricalHyperparameter(
        'activate_dropout',
        include_hyperparameter
    )
    cs.add_hyperparameter(dropout)

    # get dropout value for super_block
    # the same value will be used for
    # residual blocks in the super_blocks
    for i in range(1, super_blocks + 1):
        # for now only dropout between layers
        # 1 layer dropout for res block

        # dropout value
        dropout_value = ConfigSpace.UniformFloatHyperparameter(
            "dropout_%d_1" % i,
            lower=0,
            upper=0.7,
            default_value=0.5
        )
        cs.add_hyperparameter(dropout_value)
        # add dropout value only if active
        cs.add_condition(
            ConfigSpace.EqualsCondition(
                dropout_value,
                dropout,
                'Yes'
            )
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
    decay_type = ConfigSpace.CategoricalHyperparameter(
        'decay_type',
        decay_scheduler
    )
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
        nr_features,
        feature_type,
        max_nr_layers=19,
        nr_units=64,
        activate_dropout='No',
        activate_weight_decay='No',
        activate_batch_norm='No'
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

    include_hyperparameter = ['Yes', 'No']

    cs = ConfigSpace.ConfigurationSpace()

    # Architecture parameters
    cs.add_hyperparameter(
        ConfigSpace.Constant(
            "num_layers",
            max_nr_layers
        )
    )

    cs.add_hyperparameter(
        ConfigSpace.UniformIntegerHyperparameter(
            "batch_size",
            lower=8,
            upper=256,
            default_value=16,
            log=True
        )
    )

    # class weights and feature preprocessing
    cs.add_hyperparameter(
        ConfigSpace.CategoricalHyperparameter(
            'class_weights',
            include_hyperparameter
        )
    )

    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'feature_type',
            feature_type
        )
    )

    feature_preprocessing = ConfigSpace.CategoricalHyperparameter(
        'feature_preprocessing',
        include_hyperparameter
    )

    number_pca_components = ConfigSpace.UniformIntegerHyperparameter(
        'pca_components',
        lower=2,
        upper=nr_features,
        default_value=nr_features - 1
    )

    cs.add_hyperparameter(feature_preprocessing)
    cs.add_hyperparameter(number_pca_components)

    cs.add_condition(
        ConfigSpace.EqualsCondition(
            number_pca_components,
            feature_preprocessing,
            'Yes'
        )
    )

    # Regularition parameters
    decay_type = ConfigSpace.CategoricalHyperparameter(
        'decay_type',
        decay_scheduler
    )

    cs.add_hyperparameter(decay_type)

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
    weight_decay_activate = ConfigSpace.Constant(
        'activate_weight_decay',
        activate_weight_decay
    )

    cs.add_hyperparameter(weight_decay_activate)

    if activate_weight_decay == 'Yes':
        cs.add_hyperparameter(weight_decay)

    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'activate_batch_norm',
            activate_batch_norm
        )
    )

    dropout = ConfigSpace.Constant(
        'activate_dropout',
        activate_dropout
    )

    cs.add_hyperparameter(dropout)
    # it is the upper bound of the nr of layers,
    # since the configuration will actually be sampled.
    for i in range(1, max_nr_layers + 1):

        cs.add_hyperparameter(
            ConfigSpace.Constant(
                "num_units_%d" % i,
                nr_units
            )
        )

        dropout_value = ConfigSpace.UniformFloatHyperparameter(
            "dropout_%d" % i,
            lower=0,
            upper=0.7,
            default_value=0.5
        )

        if activate_dropout == 'Yes':
            cs.add_hyperparameter(dropout_value)

    return cs


def get_fixed_conditional_fc_config(
        nr_features, feature_type,
        max_nr_layers=19,
        nr_units=64
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

    include_hyperparameter = [
        'Yes',
        'No'
    ]

    cs = ConfigSpace.ConfigurationSpace()

    # Architecture parameters
    cs.add_hyperparameter(
        ConfigSpace.Constant(
            "num_layers",
            max_nr_layers
        )
    )

    cs.add_hyperparameter(
        ConfigSpace.UniformIntegerHyperparameter(
            "batch_size",
            lower=8,
            upper=256,
            default_value=16,
            log=True
        )
    )

    # class weights and feature preprocessing
    cs.add_hyperparameter(
        ConfigSpace.CategoricalHyperparameter(
            'class_weights',
            include_hyperparameter
        )
    )

    cs.add_hyperparameter(
        ConfigSpace.Constant(
            'feature_type',
            feature_type
        )
    )

    feature_preprocessing = ConfigSpace.CategoricalHyperparameter(
        'feature_preprocessing',
        include_hyperparameter
    )

    number_pca_components = ConfigSpace.UniformIntegerHyperparameter(
        'pca_components',
        lower=2,
        upper=nr_features,
        default_value=nr_features - 1
    )

    cs.add_hyperparameter(feature_preprocessing)
    cs.add_hyperparameter(number_pca_components)

    cs.add_condition(
        ConfigSpace.EqualsCondition(
            number_pca_components,
            feature_preprocessing,
            'Yes'
        )
    )

    # Regularization parameters
    decay_type = ConfigSpace.CategoricalHyperparameter(
        'decay_type',
        decay_scheduler
    )

    cs.add_hyperparameter(decay_type)

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

    cs.add_hyperparameter(
        ConfigSpace.CategoricalHyperparameter(
            'activate_batch_norm',
            include_hyperparameter
        )
    )

    activate_dropout = ConfigSpace.CategoricalHyperparameter(
        'activate_dropout',
        include_hyperparameter
    )

    cs.add_hyperparameter(activate_dropout)

    # it is the upper bound of the nr of layers,
    # since the configuration will actually be sampled.
    for i in range(1, max_nr_layers + 1):

        cs.add_hyperparameter(
            ConfigSpace.Constant(
                "num_units_%d" % i,
                nr_units
            )
        )

        dropout_value = ConfigSpace.UniformFloatHyperparameter(
            "dropout_%d" % i,
            lower=0,
            upper=0.7,
            default_value=0.5
        )

        cs.add_hyperparameter(dropout_value)

        cs.add_condition(
            ConfigSpace.EqualsCondition(
                dropout_value,
                activate_dropout,
                'Yes'
            )
        )

    return cs
