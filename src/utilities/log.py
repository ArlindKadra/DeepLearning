import pickle
import json
import logging
import os

import openml


def prepare_openml100(working_dir):

    # revised version of OpenML100
    benchmark_suite = openml.study.get_study("99", "tasks")
    for task in list(benchmark_suite.tasks):
        with open(os.path.join(working_dir, 'classification_tasks_100.txt'), "a") as file:
            file.write(str(task))
            file.write(" ")


def map_job_to_task(working_dir, run_id, task_id, network):

    with open(os.path.join(working_dir, 'job_to_task_mapper.txt'), "a") as file:
        file.write("%s, %s --> %d" % (network, run_id, task_id))


def general_info(working_dir, time):

    with open(os.path.join(working_dir, "results.pkl"), "rb") as fp:
        result = pickle.load(fp)

    run_id = result.get_incumbent_id()
    runs = result.get_runs_by_id(run_id)
    best_run = runs[-1]
    test_accuracy = best_run.info['test_accuracy']
    output = {}
    config_info = result.get_id2config_mapping()
    with open(os.path.join(working_dir, "best_config_info.txt"), "a") as file:

        output['test_accuracy'] = test_accuracy
        output['time'] = time
        output['config'] = config_info[run_id]
        json.dump(output, file)


def setup_logging(log_file, level=logging.INFO):
    # TODO Read main logs dir from configuration
    main_logs_dir = 'logs'
    root = logging.getLogger()
    root.setLevel(level)
    appearence_format = '%(asctime)s, %(process)-6s %(levelname)-5s %(module)s: %(message)s'

    date_format = '%H:%M:%S'

    f = logging.Formatter(appearence_format, date_format)
    ch = logging.StreamHandler()
    ch.setFormatter(f)
    root.addHandler(ch)

    os.makedirs(os.path.join(main_logs_dir), exist_ok=True)
    log_file = os.path.join(main_logs_dir, '{}.log'.format(log_file))
    fh = logging.FileHandler(log_file)
    fh.setFormatter(f)
    root.addHandler(fh)


def info_reg_methods(working_dir, network, destination_dir):

    performs_better = list()
    performs_same = 0
    performs_worse = list()
    nr_values = 0

    regularization_methods = {
        'Batch Normalization': 'resnet_only_batch',
        'Shake Shake': 'resnet_only_shake',
        'Mixup': 'resnet_only_mix',
        'Dropout': 'resnet_only_drop',
        'Weight Decay': 'resnet_only_decay'
    }

    for regularization, folder in regularization_methods.items():
        for task_id in read_task_ids(working_dir):
            try:
                with open(os.path.join(working_dir, folder,  'task_%d' % task_id, network, "best_config_info.txt"), "r") as fp:
                    output = json.load(fp)
                    regularization_value = output['test_accuracy']
                with open(os.path.join(working_dir, 'resnet_only_vanilla',  'task_%d' % task_id, network, "best_config_info.txt"), "r") as fp:
                    output = json.load(fp)
                    vanilla_value = output['test_accuracy']

                nr_values += 1
                if regularization_value > vanilla_value:
                    performs_better.append(regularization_value - vanilla_value)
                elif regularization_value == vanilla_value:
                    performs_same += 1
                else:
                    performs_worse.append(regularization_value-vanilla_value)

            except FileNotFoundError:
                # Can be that the experiment with the vanilla net
                # failed, can be the regularized one failed.
                # Comparison cannot be made.
                pass

        # finished iterating through tasks
        with open(os.path.join(os.path.expanduser(destination_dir), "info_regarding_reg_methods"), 'a') as fp:
            fp.write("%s performed better in %d tasks out of %d\n" %(regularization, len(performs_better), nr_values))
            fp.write("With mean %f, std %f\n" %(np.mean(performs_better), np.std(performs_better)))
            fp.write("Performed same in %d tasks out of %d\n" % (performs_same, nr_values))
            fp.write("Performed worse in %d tasks out of %d\n" % (len(performs_worse), nr_values))
            fp.write("With mean %f, std %f\n" % (np.mean(performs_worse), np.std(performs_worse)))
            fp.write("\n")

        nr_values = 0
        performs_worse.clear()
        performs_better.clear()
        performs_same = 0


# prepare_openml100("/home/fr/fr_fr/fr_ak547/experiments")
