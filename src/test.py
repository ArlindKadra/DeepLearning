from collections import OrderedDict
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
sns.set_style("darkgrid")

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