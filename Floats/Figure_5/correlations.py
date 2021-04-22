import pandas as pd
import scipy.stats as ss
import numpy as np
import json
import seaborn as sns
from dython.nominal import theils_u, cramers_v
from AnalysisModule.routines.util import MDefined, read_jsonfile
import ast


indf = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")

categorical_df = indf[["smiles", "elements", "bus","dimension"]]
categorical_df.rename(columns={
    "elements":"EP",
    "bus":"BUs",
    "smiles":"TA",
    "dimension": "DIM"
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
ax = sns.heatmap(heatdata_cv, annot=True, xticklabels=categorical_df.columns, yticklabels=categorical_df.columns)
ax.set_title("Cramer's V")
f.tight_layout()
f.savefig("cv.png", dpi=300)
f.savefig("cv.eps")

f, ax = plt.subplots()
ax = sns.heatmap(heatdata_tu, annot=True, xticklabels=categorical_df.columns, yticklabels=categorical_df.columns)
ax.set_title("Theil's U")
f.tight_layout()
f.savefig("tu.png", dpi=300)
f.savefig("tu.eps")

