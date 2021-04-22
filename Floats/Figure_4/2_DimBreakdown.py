import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from AnalysisModule.routines.util import read_jsonfile, count_rows_by_column1_by_unique_values_of_column2
plt.rcParams.update({'font.size': 12})

input_df = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")
Smi2Cluster = read_jsonfile("../../DataGeneration/6_AmineCluster/smiles_labeled_umap.json")
ncrystals = len(input_df)

def plot_breakdown(field="smiles", count_limit=10, savename="TA"):
    counter = Counter(input_df[field])
    dims = [0, 1, 2, 3]
    color = ["tab:blue", "tab:red", "tab:green", "tab:brown"]
    color_code = dict(zip(dims, color))
    count_df = count_rows_by_column1_by_unique_values_of_column2(input_df, col1="dimension", col2=field, cl=count_limit)
    if field == "smiles":
        count_df["cluster"] = count_df.apply(func=lambda x:Smi2Cluster[x["smiles"]], axis=1)
        count_df = count_df.sort_values(by=["cluster", "total"], ascending=[True, False])

    count_df.to_csv("2_CountDF_{}.csv".format(savename), index=False)
    xs = range(len(count_df))
    width = 0.9
    ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
    ax1.set_xticks([])
    ax1bar = count_df["total"]
    ax1.bar(xs, ax1bar, color="gray", width=0.9)
    xvline = 0
    if field == "smiles":
        smi2cluster = {k:v for k,v in Smi2Cluster.items() if k in count_df["smiles"].tolist()}
        clusters = [smi2cluster[smi] for smi in count_df["smiles"]]
        for k, v in Counter(clusters).items():
            ax1.text(xvline, max(ax1bar)*1.08, "cluster: {}".format(k), )
            xvline += v
            ax1.axvline(x=xvline-width/2, ls="dotted", c="k")

    ax1.set_ylim([0, max(ax1bar)*1.05])
    ax1.set_ylabel("Count", color="gray")
    ax1.tick_params(axis='y', labelcolor="gray")

    ax2 = plt.subplot2grid((2,2), (1,0), colspan=2)
    bars = []
    for dim in dims:
        bar = count_df[dim].tolist()
        total = count_df["total"].tolist()
        nbar = np.array([bar[i]/total[i] for i in range(len(count_df))])
        bars.append(nbar)


    for idim, dim in enumerate(dims):
        bar = bars[idim]
        c = color_code[dim]
        if idim == 0:
            ax2.bar(xs, bar, color=c,width=width, label="{}D".format(dim))
        else:
            if idim == 1:
                bottom = bars[0]
            else:
                bottom = np.zeros((idim, len(count_df)))
                for j in range(idim):
                    bottom[j] = bars[j]
                bottom = np.sum(bottom, axis=0)
            ax2.bar(xs, bar, color=c, bottom=bottom, width=width, label="{}D".format(dim))
    ax2.set_ylabel("probability", color="k")
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.set_xticks([])
    ax1.set_xlim([min(xs)-width/2, max(xs)+width/2])
    ax2.set_xlim([min(xs)-width/2, max(xs)+width/2])
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
               ncol=4, fancybox=True, shadow=False)
    plt.subplots_adjust(hspace = 0.005)
    plt.savefig('2_DimBreakdown{}.png'.format(savename), dpi=300)
    plt.savefig('2_DimBreakdown{}.eps'.format(savename))

if __name__ == '__main__':

    plot_breakdown("smiles", 10, "TA")
    plot_breakdown("elements", 10, "EP")
