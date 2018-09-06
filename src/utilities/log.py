import openml
import pickle
import os


def prepare_openml100():

    # revised version of OpenML100
    benchmark_suite = openml.study.get_study("99", "tasks")
    for task in list(benchmark_suite.tasks):
        with open("/home/kadraa/Documents/classification_tasks_100.txt", "a") as file:
            file.write(str(task))
            file.write(" ")


def save_best_config(working_dir):

    with open(os.path.join(working_dir, "results.pkl"), "rb") as fp:
        result = pickle.load(fp)

        run_id = result.get_incumbent_id()
        runs = result.get_runs_by_id(run_id)
        best_run = runs[-1]
        test_accuracy = best_run.info['test_accuracy']
        config_info = result.get_id2config_mapping()
        with open(os.path.join(working_dir, "best_config_info.txt"), "a")  as file:

                file.write("Test accuracy: ")
                file.write(str(test_accuracy))
                file.write("\n")
                file.write("Config: ")
                file.write(str(config_info[run_id]))
                file.write(" ")
