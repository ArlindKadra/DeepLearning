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

# prepare_openml100("/home/fr/fr_fr/fr_ak547/experiments")
