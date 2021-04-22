import pandas as pd
import scipy.stats as ss
import numpy as np
import json
import seaborn as sns
from dython.nominal import theils_u, cramers_v
from AnalysisModule.routines.util import MDefined, read_jsonfile
import ast


indf = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")
smiles2cluster = read_jsonfile("../../DataGeneration/6_AmineCluster/smiles_labeled_umap.json")
for cluster in [0,1,2]:
    cluster_smiles = [k for k,v in smiles2cluster.items() if v==cluster]
    categorical_df = indf[["smiles", "elements", "bus","dimension"]]
    categorical_df = categorical_df[categorical_df['smiles'].isin(cluster_smiles)]
    categorical_df.rename(columns={
        "elements":"E",
        "bus":"B",
        "smiles":"A",
        "dimension": "D"
    }, inplace=True)



    sns.set_theme()
    import matplotlib.pyplot as plt
    nfeatures = len(categorical_df.columns)
    heatdata_cv = np.zeros((nfeatures, nfeatures))
    heatdata_tu = np.zeros((nfeatures, nfeatures))

    for ic in range(nfeatures):
        col_i = categorical_df.columns[ic]
        for jc in range(nfeatures):
            col_j = categorical_df.columns[jc]
            v = cramers_v(categorical_df[col_i], categorical_df[col_j])
            u = theils_u(categorical_df[col_i], categorical_df[col_j])
            heatdata_cv[ic, jc] = round(v, 2)
            heatdata_tu[ic, jc] = round(u, 2)
            print(col_i, col_j, u)

    f, ax = plt.subplots()
    # sns.set(font_scale=2.0)
    ax.set_title("Amine cluster {}".format(cluster), fontdict={"size":18})
    ax = sns.heatmap(heatdata_tu, annot=True, xticklabels=categorical_df.columns, yticklabels=categorical_df.columns, annot_kws={"fontsize":16})
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=18)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=18)
    f.tight_layout()
    f.savefig("tu_{}.png".format(cluster), dpi=300)
    f.savefig("tu_{}.eps".format(cluster))

