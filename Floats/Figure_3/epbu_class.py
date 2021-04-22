import ast
from collections import Counter

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from AnalysisModule.routines.util import MDefined

plt.rcParams.update({'font.size': 14})


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


df = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")
elements = df["elements"]
records = []
for r in df.to_dict("records"):
    r["bus"] = ast.literal_eval(r["bus"])
    records.append(r)

classes = [ebu_class(r["elements"], r["bus"]) for r in records]
class_counter = Counter(classes)

fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))
data = list(class_counter.values())
labels = list(class_counter.keys())
labels = [labels[i] + ": {:.2%} ({})".format(data[i] / len(records), data[i]) for i in range(len(labels))]

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-90)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")
color_list = list(mcolors.TABLEAU_COLORS.keys())
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1) / 2. + p.theta1
    if ang < 90:
        ang += 90
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                horizontalalignment=horizontalalignment, color=color_list[i], **kw)
ax.set_title("Total: {}".format(len(records)), x=0.75)
plt.tight_layout()
plt.savefig("epbu_class.png", dpi=300)
plt.savefig("epbu_class.eps")
plt.clf()
