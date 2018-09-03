import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import os
import pickle
import json

sns.set()


def load_data(working_dir):

    with open(os.path.join(working_dir, "results.json"), "r") as fp:
        accuracies = {}
        val_losses = {}
        for line in fp:
            output = json.loads(line)
            budget = output[1]
            result = output[3]
            if result is not None:
                _ = result['info']
                val_loss_epochs = _['val_loss']
                test_loss = _['test_loss']
            else:
                raise ValueError("Empty Info field for the worker run")
            # append test set result for budget
            if budget not in accuracies:
                accuracies[budget] = []
            accuracies[budget].append(test_loss)
            # append val set results over epochs for budgets
            if budget not in val_losses:
                val_losses[budget] = []
            val_losses[budget].append(val_loss_epochs)
        return accuracies, val_losses



def plot_budgets_test_loss(working_dir):
    
    accuracies, val_losses = load_data(working_dir)
    fig = plt.figure(1)  # an empty figure with no axes
    # Create an axes instance
    ax = plt.subplot(111)
    data_plot = []
    for budget, values in accuracies.items():
        data_plot.append(values)
    boxplot = plt.boxplot(data_plot)
    ax.set_xticklabels(['27', '81', '243'])
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel("Budget (epochs)")
    ax.set_ylabel("Test Loss")
    ax.set_title("AutoFCResnet")
    # plt.show()
    # Save the figure
    plt.savefig(os.path.join(working_dir, 'budget_test_loss.png'), bbox_inches='tight')
    plt.close(fig)

def plot_val_loss(working_dir):

    with open(os.path.join(working_dir, "results.pkl"), "rb") as fp:
        result = pickle.load(fp)
    fig = plt.figure(2)  # an empty figure with no axes
    run_id = result.get_incumbent_id()
    runs = result.get_runs_by_id(run_id)
    best_run = runs[-1]
    # budget = best_run.budget
    validation_curve = best_run.info['val_loss']
    ax = plt.subplot(111)
    plt.plot(validation_curve)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    # test_accuracy = best_run.info['test_accuracy']
    ax.set_title("Best Hyperparameter Configuration")
    # plt.show()
    # Save the figure
    plt.savefig(os.path.join(working_dir, 'validation_curve.png'), bbox_inches='tight')
    plt.close(fig)
