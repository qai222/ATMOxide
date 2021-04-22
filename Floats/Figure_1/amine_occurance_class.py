import pandas as pd
from collections import Counter
from AnalysisModule.routines.util import read_jsonfile

input_df = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")
smi2cluster = read_jsonfile("../../DataGeneration/6_AmineCluster/smiles_labeled_umap.json")

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
fig, ax1 = plt.subplots(figsize=(7, 7))

color = 'black'
ax1.tick_params(axis='y', labelcolor=color)


def get_unique_amine_x_cumulative_y_numentries(input_df:pd.DataFrame):
    ncrystals = len(input_df)
    amine_counter = Counter(input_df["smiles"])
    namines = len(amine_counter.keys())

    cumulative_x = sorted(amine_counter.keys(), key=lambda x: amine_counter[x], reverse=True)
    ys = []
    cumulative_y = []
    nx = 0
    for x in cumulative_x:
        nx += amine_counter[x]
        ys.append(amine_counter[x])
        cumulative_y.append(nx / ncrystals)

    xs = [x / namines for x in range(namines)]
    return xs, cumulative_y, ys

colors = ["gray", "navy", "orchid", "orange"]
clusters = ["total", 0, 1, 2]
markers= ["x", "^", "o", "s"]
for cluster, color, markertype in zip(clusters, colors, markers):
    df = input_df.copy()
    if isinstance(cluster, int):
        df["smi_cluster"] = [smi2cluster[smi] for smi in input_df["smiles"]]
        df = df[df["smi_cluster"] == cluster]

    xs, c_y, ys = get_unique_amine_x_cumulative_y_numentries(df)
    # ax1.plot(xs, ys, "o:", markersize=3, label=str(cluster))
    if cluster == "total":
        lab = cluster
    else:
        lab = "cluster: {}".format(cluster)
    ax1.plot(xs, c_y, markertype, markersize=4, color=color, label=lab)

ax1.set_xlim([-0.05, 1.05])
ax1.set_xlabel('Cumulative proportion of unique amines')
ax1.set_ylabel('Cumulative proportion of structures', color="k")
# fig.legend(loc="lower right")
fig.legend(loc='lower right',
           bbox_to_anchor=(0.95, 0.2),
           )
fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig("amine_occurance_class.eps")
fig.savefig("amine_occurance_class.png", dpi=300)


