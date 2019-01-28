from collections import OrderedDict
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import spearmanr
import json
from matplotlib import colors as mcolors

import model
from fanova import fANOVA
from utilities.search_space import get_fixed_conditional_fcresnet_config
from utilities.data import determine_feature_type
import openml

sns.set_style("darkgrid")
'''
def plot_rank_correlations(working_dir):

    with open(os.path.join(working_dir, "results.pkl"), "rb") as fp:
        result = pickle.load(fp)

    low_fid = []
    high_fid = []
    runs = result.get_learning_curves()
    fig = plt.figure(4)

    # for each run a dict of config ids as keys and
    # a list of lists. The inner list contains tuples
    budgets = [9.0, 27.0, 81.0, 243.0]


    for i in range(0, len(budgets)):
        for j in range(i + 1, len(budgets)):
            small_fid = budgets[i]
            big_fidelity = budgets[j]
            for conf_id in runs.keys():
                # get the inner list
                configs = runs[conf_id][0]
                if configs[0][0] == small_fid and configs[-1][0] == big_fidelity:
                    low_fid.append(configs[0][1])
                    high_fid.append(configs[-1][1])
            plt.scatter(low_fid, high_fid)
            plt.xlabel("%.0f" % small_fid)
            plt.ylabel("%.0f" % big_fidelity)
            rco, p = spearmanr(low_fid, high_fid)
            title = "Fidelity %.0f, %.0f \n" \
                    "Rank correlation %f (p value %f)" %(small_fid, big_fidelity, rco, p)
            plt.title(title)
            low_fid.clear()
            high_fid.clear()

            plt.savefig(
                os.path.join(
                    working_dir,
                    'fidelity%f%f.pdf' %(small_fid, big_fidelity)
                ),
                bbox_inches = 'tight'
            )
            plt.clf()
    plt.close(fig)

plot_rank_correlations('C:\\Users\\Lindarx\\Desktop\\thesis info\\experiments_fcresnet_vanilla_fixed\\task_3\\fcresnet')
'''
def read_task_ids(working_dir):

    task_ids = []
    with open(os.path.join(working_dir, "classification_tasks_100.txt"), "r") as fp:

        # there is an illegal value for the int
        # cast if the last space is not removed
        line = fp.readline().rstrip()
        for number in line.split(" "):
            task_ids.append(int(number))

    return task_ids


def plot_reg_methods_over_vanilla(working_dir, network, destination_dir):


    methods = {
        'Vanilla network': 'resnet_only_vanilla',
        'Network + Batch Normalization': 'resnet_only_batch',
        'Network + Shake Shake': 'resnet_only_shake',
        'Network + Mixup': 'resnet_only_mix',
        'Network + Dropout': 'resnet_only_drop',
        'Network + Weight Decay': 'resnet_only_decay'
    }

    colors = {
        'Vanilla network': '#747dc9',
        'Network + Batch Normalization': '#7aa444',
        'Network + Shake Shake': '#b65cbf',
        'Network + Mixup': '#4aad91',
        'Network + Dropout': '#ca5670',
        'Network + Weight Decay': '#c57c3c'
    }

    phases = ['train', 'validation']

    for task_id in read_task_ids(working_dir):
        fig = plt.figure(task_id)  # an empty figure with no axes
        for phase in phases:
            for method, folder in methods.items():
                try:
                    with open(os.path.join(working_dir, folder, 'task_%d' % task_id, network, "results.pkl"), "rb") as fp:
                        result = pickle.load(fp)
                    run_id = result.get_incumbent_id()
                    runs = result.get_runs_by_id(run_id)
                    best_run = runs[-1]
                    info = best_run.info
                    budget = int(info['nr_epochs'])
                    epochs = [x for x in range(1, budget + 1)]
                    validation_acc = info['val_accuracy']
                    validation_error_rates = np.ones(len(validation_acc)) * 100 - validation_acc
                    train_curve = info['train_loss']
                    ax = plt.subplot(111)
                    if phase == 'train':
                        plt.plot(epochs, train_curve, label=method, color=colors[method])
                    elif phase == 'validation':
                        plt.plot(epochs, validation_error_rates, label=method, color=colors[method])
                except:
                    # No pickle file for this method
                    pass

            ax.set_xlabel("Epoch")
            if phase == 'train':
                ax.set_ylabel("Loss")
            else:
                ax.set_ylabel("Error rate")
            ax.set_title("Comparison of Incumbents")
            ax.legend(loc='best')
            path = os.path.join(os.path.expanduser(destination_dir), 'comparison_incumbent_error_rate', str(task_id))
            if os.path.exists(path):
                if not os.path.isdir(path):
                    os.makedirs(path)
            else:
                os.makedirs(path)
            plt.savefig(os.path.join(path, '%s_curve.pdf' %(phase)), bbox_inches='tight')
            plt.clf()

    plt.close(fig)

# plot_reg_methods_over_vanilla('/home/kadraa/Documents/old_results', 'fcresnet', '/home/kadraa/Documents/old_results')
def calculate_dataset_sizes(task_ids):

    tasks = openml.tasks.get_tasks(task_ids)
    dataset_ids = [task.dataset_id for task in tasks]
    datasets = openml.datasets.get_datasets(dataset_ids)
    dataset_sizes = [dataset.qualities["NumberOfInstances"] for dataset in datasets]
    return np.array(dataset_sizes)


def calculate_reg_method_stat(working_dir, network, destination_dir):

    abs_performs_better = list()
    rel_performs_better = list()
    best_tasks = list()
    abs_performs_same = 0
    rel_performs_same = 0
    same_tasks = list()
    abs_performs_worse = list()
    rel_performs_worse = list()
    worst_tasks = list()
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
                    abs_performs_better.append(regularization_value - vanilla_value)
                    rel_performs_better.append((regularization_value - vanilla_value) / vanilla_value)
                    best_tasks.append(task_id)
                elif regularization_value == vanilla_value:
                    abs_performs_same += 1
                    rel_performs_same +=1
                    same_tasks.append(task_id)
                else:
                    abs_performs_worse.append(regularization_value - vanilla_value)
                    rel_performs_worse.append((regularization_value - vanilla_value) / vanilla_value)
                    worst_tasks.append(task_id)

            except FileNotFoundError:
                # Can be that the experiment with the vanilla net
                # failed, can be the regularized one failed.
                # Comparison cannot be made.
                pass

        path = os.path.join(os.path.expanduser(destination_dir))
        if os.path.exists(path):
            if not os.path.isdir(path):
                os.makedirs(path)
        else:
            os.makedirs(path)

        all_tasks_abs = list()
        all_tasks_abs.extend(abs_performs_better)
        all_tasks_abs.extend([0] * (abs_performs_same))
        all_tasks_abs.extend(abs_performs_worse)
        all_tasks_rel = list()
        all_tasks_rel.extend(rel_performs_better)
        all_tasks_rel.extend([0] * (rel_performs_same))
        all_tasks_rel.extend(rel_performs_worse)
        # finished iterating through tasks
        with open(os.path.join(os.path.expanduser(destination_dir), "reg_method_info"), 'a') as fp:

            rel_performs_better = np.array(rel_performs_better)
            best_task_sizes = calculate_dataset_sizes(best_tasks)
            worst_task_sizes = calculate_dataset_sizes(worst_tasks)
            fp.write("%s performed better in %d tasks out of %d\n" %(regularization, len(abs_performs_better), nr_values))
            fp.write("Average dataset size %d, largest dataset %d, smallest dataset %d\n" %(np.mean(best_task_sizes), np.max(best_task_sizes), np.min(best_task_sizes)))
            fp.write("With mean absolute improvement %f and std %f, geometric mean relative improvement %f\n" %(np.mean(abs_performs_better), np.std(abs_performs_better), rel_performs_better.prod() ** (1.0 / len(rel_performs_better)) ))
            fp.write("Performed same in %d tasks out of %d\n" % (abs_performs_same, nr_values))
            fp.write("Performed worse in %d tasks out of %d\n" % (len(abs_performs_worse), nr_values))
            fp.write("Average dataset size %d, largest dataset %d, smallest dataset %d\n" % (np.mean(worst_task_sizes), np.max(worst_task_sizes), np.min(worst_task_sizes)))
            fp.write("With mean absolute decrease %f and std %f\n" % (np.mean(abs_performs_worse), np.std(abs_performs_worse)))
            fp.write("Mean absolute improvement over all tasks %f\n" %(np.mean(all_tasks_abs)))
            all_tasks_rel = np.array(all_tasks_rel)
            fp.write("Mean relative improvement over all tasks %f\n" % (np.mean(all_tasks_rel)))
            fp.write("\n")

        abs_performs_better.clear()
        rel_performs_better = list()
        best_tasks.clear()
        abs_performs_same = 0
        rel_performs_same = 0
        same_tasks.clear()
        abs_performs_worse.clear()
        rel_performs_worse.clear()
        worst_tasks.clear()
        nr_values = 0

# calculate_reg_method_stat('/home/kadraa/Documents/old_results', 'fcresnet', '/home/kadraa/Documents/old_results')

def calculate_batchnorm_lr(working_dir, network, destination_dir):

    high_learning_rates = list()
    all_learning_rates = list()
    improvement_learning_rates = list()
    methods = {
        'Vanilla network': 'resnet_only_vanilla',
        'Network + Shake Shake': 'resnet_only_shake',
        'Network + Mixup': 'resnet_only_mix',
        'Network + Dropout': 'resnet_only_drop',
        'Network + Weight Decay': 'resnet_only_decay'
    }

    for method, folder in methods.items():
        for task_id in read_task_ids(working_dir):
            try:
                with open(os.path.join(working_dir, folder,  'task_%d' % task_id, network, "best_config_info.txt"), "r") as fp:
                    output = json.load(fp)
                    method_learning_rate = output["config"]["config"]["learning_rate"]
                with open(os.path.join(working_dir, 'resnet_only_batch',  'task_%d' % task_id, network, "best_config_info.txt"), "r") as fp:
                    output = json.load(fp)
                    batch_learning_rate = output["config"]["config"]["learning_rate"]

                all_learning_rates.append(batch_learning_rate)
                if batch_learning_rate > method_learning_rate:
                    high_learning_rates.append(batch_learning_rate)
                    improvement_learning_rates.append((batch_learning_rate - method_learning_rate) / method_learning_rate)

            except FileNotFoundError:
                # Can be that the experiment with the vanilla net
                # failed, can be the regularized one failed.
                # Comparison cannot be made.
                pass

        path = os.path.join(os.path.expanduser(destination_dir))
        if os.path.exists(path):
            if not os.path.isdir(path):
                os.makedirs(path)
        else:
            os.makedirs(path)

        # finished iterating through tasks
        with open(os.path.join(os.path.expanduser(destination_dir), "batch_norm_lr_info"), 'a') as fp:

            fp.write("Number of times the learning rate was higher than %s: %d from %d\n" % (method, len(high_learning_rates), len(all_learning_rates)))
            fp.write("Mean learning rate %f for the cases that surpasses %s, mean relative improvement %f\n" %(np.mean(high_learning_rates), method, np.mean(improvement_learning_rates)))
            fp.write("Mean learning rate %f for all cases\n" % (np.mean(all_learning_rates)))
            fp.write("\n")

        high_learning_rates.clear()
        all_learning_rates.clear()

# calculate_batchnorm_lr('/home/kadraa/Documents/old_results', 'fcresnet', '/home/kadraa/Documents/old_results')

def calculate_performance_over_dataset_size(working_dir, network, destination_dir):

    task_dataset_sizes = list()
    performance_tasks = list()

    default_task_ids = read_task_ids(working_dir)
    default_task_dataset_sizes = \
        calculate_dataset_sizes(default_task_ids)

    dataset_size_for_task = dict()

    for task, task_size in zip(default_task_ids, default_task_dataset_sizes):
        dataset_size_for_task[task] = task_size



    methods = {
        'Vanilla network': 'resnet_only_vanilla',
        'Network + Shake Shake': 'resnet_only_shake',
        'Network + Mixup': 'resnet_only_mix',
        'Network + Dropout': 'resnet_only_drop',
        'Network + Weight Decay': 'resnet_only_decay',
        'Network + Conditional Reg': 'resnet_only_conditional'
    }

    path = os.path.join(os.path.expanduser(destination_dir), "performance_over_dataset_size")
    if os.path.exists(path):
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        os.makedirs(path)

    for method, folder in methods.items():
        fig = plt.figure(method)
        for task_id in read_task_ids(working_dir):
            try:
                with open(os.path.join(working_dir, folder,  'task_%d' % task_id, network, "best_config_info.txt"), "r") as fp:
                    output = json.load(fp)
                performance_tasks.append(output['test_accuracy'])
                task_dataset_sizes.append(dataset_size_for_task[task_id])
            except:
                # No pickle file for this method
                pass


        plt.scatter(task_dataset_sizes, performance_tasks)
        ax = plt.subplot(111)
        ax.set_xlabel("Dataset size")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Performance of %s over dataset size" %(method))
        # ax.legend(loc='best')
        plt.savefig(os.path.join(path, '%s.pdf' % (method)), bbox_inches='tight')
        plt.clf()

        # changing method
        performance_tasks.clear()
        task_dataset_sizes.clear()

    plt.close(fig)

# calculate_performance_over_dataset_size('/home/kadraa/Documents/exp1-thesis', 'fcresnet', '/home/kadraa/Documents/exp1-thesis')

def calculate_hp_importance_over_dataset_size(working_dir, network, destination_dir):

    task_dataset_sizes = list()

    #default_task_ids = read_task_ids(working_dir)
    #default_task_dataset_sizes = \
        #calculate_dataset_sizes(default_task_ids)

    #dataset_size_for_task = dict()

    #for task, task_size in zip(default_task_ids, default_task_dataset_sizes):
      #  dataset_size_for_task[task] = task_size



    methods = {
        'Conditional Network': 'resnet_only_conditional',
        'Network + Shake Shake': 'resnet_only_shake',
        'Network + Mixup': 'resnet_only_mix',
        'Network + Dropout': 'resnet_only_drop',
        'Network + Weight Decay': 'resnet_only_decay'
    }

    config_spaces = {
        'Conditional Network': None,
        'Network + Shake Shake': {'activate_shake_shake': 'Yes'},
        'Network + Mixup': {'activate_mixout': 'Yes'},
        'Network + Dropout': {'activate_dropout': 'Yes'},
        'Network + Weight Decay': {'activate_weight_decay': 'Yes'}
    }


    for method, folder in methods.items():
        fig = plt.figure(method)
        for task_id in read_task_ids(working_dir):
            try:
                with open(os.path.join(working_dir, folder, 'task_%d' % task_id, network, "results.pkl"), "rb") as fp:
                    result = pickle.load(fp)
                model.Loader(task_id)
                x, _, categorical = model.get_dataset()
                feature_type = determine_feature_type(categorical)
                nr_features = x.shape[1]
                if config_spaces[method] is None:
                    x, y, config = result.get_fANOVA_data(get_fixed_conditional_fcresnet_config(nr_features, feature_type))
                else:
                    x, y, config = result.get_fANOVA_data(get_fixed_conditional_fcresnet_config(nr_features, feature_type, **config_spaces[method]))
                np.place(x, x==None, [0, 0.1])
                np.place(x, x =='Yes', [1])
                np.place(x, x == 'No', [0])

                fanova_object = fANOVA(x, y)
                best_marginals = fanova_object.get_most_important_pairwise_marginals(5)
                print(best_marginals)
                # task_dataset_sizes.append(dataset_size_for_task[task_id])
            except Exception as e:
                # No pickle file for this method
                raise e
                pass
        """
        path = os.path.join(os.path.expanduser(destination_dir), "%s" %(method), "hp_importance")
        if os.path.exists(path):
            if not os.path.isdir(path):
                os.makedirs(path)
        else:
            os.makedirs(path)

        plt.scatter(task_dataset_sizes, performance_tasks)
        ax = plt.subplot(111)
        ax.set_xlabel("Dataset size")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Performance of %s over dataset size" %(method))
        # ax.legend(loc='best')
        plt.savefig(os.path.join(path, '%s.pdf' % (method)), bbox_inches='tight')
        plt.clf()

        # changing method
        performance_tasks.clear()
        task_dataset_sizes.clear()

        plt.close(fig)
        """
calculate_hp_importance_over_dataset_size('/home/kadraa/Documents/exp1-thesis', 'fcresnet', '/home/kadraa/Documents/exp1-thesis')