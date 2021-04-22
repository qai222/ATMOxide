import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

from collections import Counter
"""
each P is associated with a count, which is the sample size
one can set a count limit s.t. P with count lower than that limit are discarded
plot the count lim against the number of structures *remained*
"""

input_df = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")
ncrystals = len(input_df)

def CountLimPlot(field="smiles", ylable="TA", mark_x=10, ):
    counter = Counter(input_df[field])
    unique_values = sorted(set(input_df[field]), key=lambda x:counter[x], reverse=True)
    xs = []
    ys0 = []  # # of unique amines
    ys1 = []  # # of structures
    for lim in range(1,20):
        xs.append(lim)
        y0 = len([smi for smi in unique_values if counter[smi] >= lim])
        y1 = sum([counter[smi] for smi in unique_values if counter[smi] >= lim])
        ys0.append(y0)
        ys1.append(y1/ncrystals)

    fig, ax1 = plt.subplots()
    ipt = mark_x - 1
    color = 'tab:red'
    ax1.set_xlabel('Minmum sample size')
    ax1.set_ylabel('Number of unique {}'.format(ylable), color=color)  # we already handled the x-label with ax1
    ax1.plot(xs, ys0, ":", color=color)
    ax1.set_xticks(xs[1::2])
    ax1.annotate("  {}".format(ys0[ipt]), (xs[ipt], ys0[ipt]), color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Proportion of structures', color=color)
    ax2.scatter(xs, ys1, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axvline(xs[ipt],color="k", ls=":")
    ax2.annotate("  {:.1f}% ({} structures)".format(ys1[ipt]*100, int(ys1[ipt]*ncrystals)), (xs[ipt], ys1[ipt]), color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig("1_CountLim{}.png".format(ylable), dpi=300)
    fig.savefig("1_CountLim{}.eps".format(ylable))

if __name__ == '__main__':

    CountLimPlot("smiles", "amine", 10)
    CountLimPlot("elements", "elements", 10)
