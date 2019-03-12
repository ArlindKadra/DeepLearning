from collections import OrderedDict
from collections import defaultdict
import os
import pickle
import math
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import json
from matplotlib import colors as mcolors
import ConfigSpace as CS

import model
from fanova import fANOVA
from utilities.search_space import get_fixed_conditional_fanova_fcresnet_config
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

def string_to_numerical_categorical(hyperparameter, value):

    optimizers = {
        'Adam': 0,
        'AdamW': 1,
        'SGD': 2,
        'SGDW': 3
    }

    include = {
        'Yes': 0,
        'No': 1
    }

    decay_scheduler = {
        'cosine_annealing': 0,
        'cosine_decay': 1,
        'exponential_decay': 2
    }

    other_categorical_hyper = {
        'class_weights',
        'feature_preprocessing',
        'mixout',
        'shake-shake',
        'activate_batch_norm',
        'activate_weight_decay',
        'activate_dropout'
    }

    if hyperparameter == 'optimizer':
        return optimizers[value]
    elif hyperparameter == 'decay_type':
        return decay_scheduler[value]
    elif hyperparameter in other_categorical_hyper:
        return include[value]
    else:
        return value

def calculate_hp_importance_over_dataset_size(working_dir, network, destination_dir):

    budgets = [81]

    importance = defaultdict(lambda: list())

    default_task_ids = read_task_ids(working_dir)
    default_task_dataset_sizes = \
        calculate_dataset_sizes(default_task_ids)
    dataset_size_for_task = dict()

    for task, task_size in zip(default_task_ids, default_task_dataset_sizes):
        dataset_size_for_task[task] = task_size

    fig = plt.figure(5)

    methods = {
        'Conditional Network': 'resnet_only_conditional',
    }
    task_dataset_sizes = list()
    config_space = get_fixed_conditional_fanova_fcresnet_config()
    hp_names = list(map(lambda hp: hp.name, config_space.get_hyperparameters()))
    for method, folder in methods.items():
        for task_id in read_task_ids(working_dir):

            X = []
            y = []
            try:
                with open(os.path.join(working_dir, folder, 'task_%d' % task_id, network, "results.pkl"), "rb") as fp:
                    result = pickle.load(fp)

                id2conf = result.get_id2config_mapping()
                all_runs = result.get_all_runs(only_largest_budget=False)
                all_runs = list(filter(lambda r: r.budget in budgets, all_runs))

                for r in all_runs:
                    if r.loss is None: continue
                    config = id2conf[r.config_id]['config']
                    X.append([string_to_numerical_categorical(hp, config[hp]) for hp in hp_names])
                    y.append(r.loss)

            except FileNotFoundError:
                continue

            if len(X) > 0:
                fanova_object = fANOVA(np.asarray(X), np.asarray(y), config_space)
            else:
                continue
            for hp in hp_names:
                importance_hp = fanova_object.quantify_importance((hp,)).get((hp,))
                importance[hp].append(importance_hp['individual importance'])
            task_dataset_sizes.append(dataset_size_for_task[task_id])
        path = os.path.join(os.path.expanduser(destination_dir), 'importance_over_datasets', method)
        if os.path.exists(path):
            if not os.path.isdir(path):
                os.makedirs(path)
        else:
            os.makedirs(path)

        plt.xlabel("Dataset size")
        for hp in importance:

            plt.scatter(task_dataset_sizes, importance[hp])
            plt.ylabel("Importance %s" % hp)
            plt.rcParams['axes.unicode_minus'] = False
            title = "Importance of  %s over dataset size" % hp
            plt.title(title)
            plt.savefig(
                os.path.join(
                    path,
                    'importance_%s.pdf' % hp
                ),
                bbox_inches='tight'
            )
            plt.clf()
    plt.close(fig)

# calculate_hp_importance_over_dataset_size('/home/kadraa/Documents/exp1-thesis', 'fcresnet', '/home/kadraa/Documents/exp1-thesis')



def plot_rank_correlations_over_dataset_size(working_dir, network, method, destination_dir):

    fidelities = defaultdict(lambda :list())

    default_task_ids = read_task_ids(working_dir)
    default_task_dataset_sizes = \
        calculate_dataset_sizes(default_task_ids)

    fig = plt.figure(4)

    for task_id in read_task_ids(working_dir):

        try:
            with open(os.path.join(working_dir, method, 'task_%d' % task_id, network, "results.pkl"), "rb") as fp:
                result = pickle.load(fp)
        except:
            # no pickle for this task, pass
            pass

        low_fid = []
        high_fid = []
        runs = result.get_learning_curves()

        budgets = [30.0, 60.0, 120.0, 240.0]
        for i in range(0, len(budgets)):
            small_fid = budgets[i]
            for j in range(i + 1, len(budgets)):
                big_fidelity = budgets[j]
                # for each run a dict of config ids as keys and
                # a list of lists. The inner list contains tuples
                for conf_id in runs.keys():
                    # get the inner list
                    configs = runs[conf_id][0]
                    if configs is not None and len(configs) > 0:
                        if configs[0][0] == small_fid and configs[-1][0] == big_fidelity:
                            low_fid.append(configs[0][1])
                            high_fid.append(configs[-1][1])

                rco, p = spearmanr(low_fid, high_fid)
                fidelity_comb_name = '%s-%s' % (int(small_fid), int(big_fidelity))
                fidelities[fidelity_comb_name].append((rco, p))

        path = os.path.join(os.path.expanduser(destination_dir), 'rc_over_datasets', method)
        if os.path.exists(path):
            if not os.path.isdir(path):
                os.makedirs(path)
        else:
            os.makedirs(path)

        temp_x = []
        temp_y = []
        for fidelity_combination in fidelities:
            task_counter = 0
            for task_value in fidelities[fidelity_combination]:
                temp_x.append(default_task_dataset_sizes[task_counter])
                # the first element is the rco
                temp_y.append(task_value[0])
                task_counter += 1

            plt.scatter(temp_x, temp_y)
            plt.xlabel("Dataset size")
            plt.ylabel("Rank correlation %s" % fidelity_combination)
            plt.rcParams['axes.unicode_minus'] = False
            title = "Rank correlation %s over dataset size" % fidelity_combination
            plt.title(title)
            temp_x.clear()
            temp_y.clear()

            plt.savefig(
                os.path.join(
                    path,
                    'rankcorrelation_%s.pdf' % fidelity_combination
                ),
                bbox_inches='tight'
            )
            plt.clf()
    plt.close(fig)

    with open(os.path.join(path, 'averaged_rc'), 'a') as fp:
        for fidelity_combination in fidelities:
            rc_fidelity_comb = [rc for rc, _ in fidelities[fidelity_combination] if not math.isnan(rc)]
            p_fidelity_comb = [p for _, p in fidelities[fidelity_combination] if not math.isnan(p)]
            fp.write('Rank correlaton %s, mean value %.3f, std %.3f\n'
                     % (fidelity_combination, np.mean(rc_fidelity_comb), np.std(rc_fidelity_comb)))
            fp.write('P value %s, mean value %.3f, std %.3f\n'
                     % (fidelity_combination, np.mean(p_fidelity_comb), np.std(p_fidelity_comb)))

plot_rank_correlations_over_dataset_size('/home/kadraa/Documents/final_exp_thesis', 'fcresnet', 'resnet_only_cond64v2', '/home/kadraa/Documents/final_exp_thesis')