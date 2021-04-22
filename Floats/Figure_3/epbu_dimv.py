import ast
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 12})
from collections import Counter

import pandas as pd

from AnalysisModule.routines.util import MDefined, count_rows_by_column1_by_unique_values_of_column2

df = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")
elements = df["elements"]
records = []
for r in df.to_dict("records"):
    r["bus"] = ast.literal_eval(r["bus"])
    records.append(r)

def ebu_class(elements: str, bus: list):
    elements = ast.literal_eval(elements)
    mset = set(elements).intersection(MDefined)
    # MOx no BU
    if len(mset) == 1 and len(bus) == 0:
        return r"MO$x$ 0PBU"
    if len(mset) > 1 and len(bus) == 0:
        return r"M$y$O$x$ 0PBU"
    if len(mset) == 1 and len(bus) == 1:
        return r"MO$x$ 1PBU"
    if len(mset) == 1 and len(bus) > 1:
        return r"MO$x$ $z$PBU"
    if len(mset) > 1 and len(bus) == 1:
        return r"M$y$O$x$ 1PBU"
    if len(mset) > 1 and len(bus) > 1:
        return r"M$y$O$x$ $z$PBU"
    return "unknown"

classes = [ebu_class(r["elements"], r["bus"]) for r in records]
df["epbu_class"] = classes

count_df = count_rows_by_column1_by_unique_values_of_column2(df, col1="dimension", col2="epbu_class")
prepend_total_df = pd.DataFrame()
prepend_total_df["epbu_class"] = ["total"]
prepend_total = 0
for i in [0,1,2,3]:
    prepend_total_df[i] = sum(count_df[i])
    prepend_total += prepend_total_df[i]
prepend_total_df["total"] = len(df)
count_df = prepend_total_df.append(count_df)
print(count_df)


dims = [0, 1, 2, 3]
color = ["tab:blue", "tab:red", "tab:green", "tab:brown"]
color_code = dict(zip(dims, color))

xs = list(range(len(count_df)))
width = 0.9
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax1.set_xticks([])
# ax1.axis('off')
ax1bar = count_df["total"].tolist()
ax1.bar(xs, ax1bar, color="gray", width=0.9)
ax1.set_ylim([0, max(ax1bar)])
ax1.set_ylabel("count", color="gray")
ax1.tick_params(axis='y', labelcolor="gray")

ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
bars = []
for dim in dims:
    bar = count_df[dim].tolist()
    total = count_df["total"].tolist()
    nbar = np.array([bar[i] / total[i] for i in range(len(count_df))])
    bars.append(nbar)

for idim, dim in enumerate(dims):
    bar = bars[idim]
    c = color_code[dim]
    if idim == 0:
        ax2.bar(xs, bar, color=c, width=width, label="{}D".format(dim))
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
ax2.set_xticks(xs)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.set_xticklabels(count_df["epbu_class"].tolist(), rotation=-30)
ax1.set_xlim([min(xs) - width / 2, max(xs) + width / 2])
ax2.set_xlim([min(xs) - width / 2, max(xs) + width / 2])
ax2.set_ylim([0, 1])
ax2.legend(loc='lower center', bbox_to_anchor=(0.95, 1.5),
           ncol=1, fancybox=True, shadow=False)
plt.subplots_adjust(hspace=0.005)
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('epbu_dimv.eps')
plt.savefig('epbu_dimv.png', dpi=300)
