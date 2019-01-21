from collections import OrderedDict
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr

def plot_rank_correlations(working_dir):

    with open(os.path.join(working_dir, "results.pkl"), "rb") as fp:
        result = pickle.load(fp)

    values = OrderedDict()
    fig = plt.figure(4)  # an empty figure with no axes
    runs = result.get_learning_curves()

    iterations = list(set([len(value[0]) for value in runs.values()]))
    # for each run a dict of config ids as keys and
    # a list of lists. The inner list contains tuples
    budgets = {9: 0, 27: 1, 81: 2, 243: 3}

    for conf_id in runs.keys():
        # get the inner list
        configs = runs[conf_id][0]
        if len(configs) == 2:



'''

    fidelities = np.zeros((4, 4))
    budgets = list(high_fids.keys())
    budgets.sort()

    for i in range(0, len(fidelities)):
        for j in range(i + 1, len(fidelities)):
            if j == 1:
                sample_from = low_fids
            elif j == 2:
                sample_from = medium_fids
            elif j == 3:
                sample_from = high_fids
            low_fid = sample_from[budgets[i]]
            high_fid = sample_from[budgets[j]]
            rho, _ = spearmanr(low_fid, high_fid)
            fidelities[i][len(fidelities) - j -1] = rho





    mask = np.zeros_like(fidelities)
    # remove redudant information since the matrix is symmetrical
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(
            fidelities, mask=mask, vmax=1,
            annot=True, yticklabels=budgets,
            xticklabels=budgets, square=True,
            cmap="YlGnBu"
        )
        ax.set_title("Rank correlation")
        '''
''''
        plt.savefig(
            os.path.join(
                working_dir,
                'rank_correlations.pdf'
            ),
            bbox_inches = 'tight'
        )
        plt.close(fig)
'''

plot_rank_correlations('/home/kadraa/Documents/updated/experiments_fcresnet_vanilla_fixed/task_3/fcresnet')