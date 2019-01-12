import os
import pickle
import numpy as np

from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from utilities.data import determine_feature_type
from utilities.search_space import (
    get_fc_config,
    get_fixed_fcresnet_config,
    get_fixed_fc_config,
    get_fixed_conditional_fc_config,
    get_super_fcresnet_config,
    get_fixed_conditional_fcresnet_config
)
import openml_experiment
import model
import config as configuration
import utilities.data
import utilities.regularization


class Master(object):

    def __init__(
            self,
            num_workers,
            num_iterations,
            run_id,
            array_id,
            working_dir,
            nic_name,
            network,
            min_budget,
            max_budget,
            eta
    ):

        x, y, categorical = model.get_dataset()
        feature_type = determine_feature_type(categorical)
        nr_features = x.shape[1]

        if network == 'fcresnet':
            config_space = get_fixed_fcresnet_config(
                nr_features,
                feature_type,
                num_res_blocks=4
            )
        else:
            config_space = get_fixed_conditional_fc_config(
                nr_features,
                feature_type,
                max_nr_layers=5
            )

        if array_id == 1:

            result_logger = hpres.json_result_logger(directory=working_dir, overwrite=True)

            # start nameserver
            ns = hpns.NameServer(run_id=run_id, nic_name=nic_name,
                                 working_directory=working_dir)

            ns_host, ns_port = ns.start()  # stores information for workers to find in working_directory

            # BOHB is usually so cheap, that we can affort to run a worker on the master node, too.
            worker = Slave(nameserver=ns_host, nameserver_port=ns_port, run_id=run_id)
            worker.run(background=True)

            hb = BOHB(
                configspace=config_space,
                run_id=run_id,
                eta=eta,
                min_budget=min_budget,
                max_budget=max_budget,
                host=ns_host,
                nameserver=ns_host,
                result_logger=result_logger,
                nameserver_port=ns_port,
                ping_interval=3600
            )

            # BOHB can wait until a minimum number of workers
            # is online before starting
            res = hb.run(
                n_iterations=num_iterations,
                min_n_workers=num_workers
            )

            # pickle result here for later analysis
            with open(os.path.join(working_dir, 'results.pkl'), 'wb') as fh:
                pickle.dump(res, fh)

            # shutdown all workers
            hb.shutdown(shutdown_workers=True)

            # and the nameserver
            ns.shutdown()

        else:

            host = hpns.nic_name_to_host(nic_name)

            # workers only instantiate the Slave, find the nameserver and start serving
            w = Slave(run_id=run_id, host=host)
            w.load_nameserver_credentials(working_dir)
            # run worker in the forground,
            w.run(background=False)


class Slave(Worker):

    def compute(self, config, budget, *args, **kwargs):
        """All the functionality that the worker will compute.

        The worker will train the neural network and
        save results related to the performance.

        Args:
            config: A hyperparameter configuration drawn from the ConfigSpace.
            budget: budget on which the training of the network will be limited.
        """
        x, y, _ = model.get_dataset()
        train_indices, test_indices = model.get_split_indices()

        hardcoded_folds = \
            {
                9: 4,
                27: 6,
                91: 8,
                243: 10,
            }
        epochs = int(budget)

        # the budget is the number of epochs
        if configuration.fidelity == 'epochs':
            nr_folds = 10
        # the budget is the number of epochs
        # folds also given as fidelity

        elif configuration.fidelity == 'both':
            nr_folds = hardcoded_folds[epochs]

        if configuration.cross_validation:
            output = utilities.regularization.cross_validation(
                epochs,
                x,
                y,
                config,
                train_indices,
                test_indices,
                nr_folds=nr_folds
            )
            val_accuracy = output['val_accuracy']

        else:
            x_train_split = x[train_indices]
            y_train_split = y[train_indices]
            training_indices, validation_indices = \
                utilities.data.determine_stratified_val_set(
                    x_train_split,
                    y_train_split
                )
            set_indices = (
                training_indices,
                validation_indices,
                test_indices
            )
            output = openml_experiment.train(
                config,
                configuration.network_type,
                epochs,
                x,
                y,
                set_indices
            )

            val_loss_epochs = output['validation'][0]
            if isinstance(val_loss_epochs, np.ndarray):
                val_loss_epochs = list(val_loss_epochs)
            val_accuracy = output['validation'][1]
            train_loss_epochs = output['train']
            test_loss = output['test'][0]
            test_accuracy = output['test'][1]
            output = {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'val_loss': val_loss_epochs,
                'train_loss': list(train_loss_epochs),
                'nr_epochs': epochs
            }

        if configuration.predictive_measure == 'loss':

            val_loss = output["val_loss"]
            # check if it is a list
            if isinstance(val_loss, list):
                result_measure = val_loss[-1]
            else:
                result_measure = val_loss
        elif configuration.predictive_measure == 'error_rate':

            if isinstance(val_accuracy, list):
                success_rate = val_accuracy[-1]
            else:
                success_rate = val_accuracy
            # it is in % so dividing by 100
            result_measure = 1 - (success_rate / 100)

        return ({
            'loss': float(result_measure),
            'info': output
        })
