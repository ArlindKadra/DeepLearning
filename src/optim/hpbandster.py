import model
from utilities.data import shuffle_data, separate_input_sets
from utilities.regularization import cross_validation
from models import fcresnet

import numpy as np
import os
import pickle
from hpbandster.optimizers import BOHB
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker


class Master(object):

    def __init__(self, num_workers, num_iterations, run_id, array_id, working_dir, nic_name):

        config_space = fcresnet.get_config_space()
        if array_id == 1:

            result_logger = hpres.json_result_logger(directory=working_dir, overwrite=True)

            # start nameserver
            ns = hpns.NameServer(run_id=run_id, nic_name=nic_name,
                                 working_directory=working_dir)

            ns_host, ns_port = ns.start()  # stores information for workers to find in working_directory

            # BOHB is usually so cheap, that we can affort to run a worker on the master node, too.
            worker = Slave(nameserver=ns_host, nameserver_port=ns_port, run_id=run_id)
            worker.run(background=True)

            hb = BOHB(configspace=config_space,
                      run_id=run_id,
                      eta=3, min_budget=9, max_budget=243,
                      host=ns_host,
                      nameserver=ns_host,
                      result_logger=result_logger,
                      nameserver_port=ns_port,
                      ping_interval=3600
                      )

            res = hb.run(n_iterations=num_iterations,
                         min_n_workers=num_workers
                         # BOHB can wait until a minimum number of workers is online before starting
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

    def compute(self, config, budget, k_fold_validation=False, *args, **kwargs):
        """All the functionality that the worker will compute.

        The worker will train the neural network and
        save results related to the performance.

        Args:
            config: A hyperparameter configuration drawn from the ConfigSpace.
            budget: budget on which the training of the network will be limited.
            k_fold_validation: Flag to control cross validation.
        """
        x, y, _ = model.get_dataset()

        if k_fold_validation:
            output = cross_validation(int(budget), x, y, config)
        else:
            x, y = shuffle_data(x, y)

            examples, labels = separate_input_sets(x, y)

            output = fcresnet.train(
                config,
                int(budget),
                examples['train'],
                labels['train'],
                examples['val'],
                labels['val'],
                examples['test'],
                labels['test']
            )

            val_loss_epochs = output['validation']
            test_loss = output['test'][0]
            test_accuracy = output['test'][1]
            output = {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'val_loss': list(val_loss_epochs)
            }

        val_loss = output["val_loss"]

        return ({
            'loss': (np.mean(val_loss)).item(),  # this is the a mandatory field to run hyperband
            'info': output  # can be used for any user-defined information - also mandatory
        })
