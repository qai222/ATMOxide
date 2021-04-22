import pandas as pd
from pprint import pprint
from AnalysisModule.prepare.diagram import BuildingUnit
from AnalysisModule.routines.util import read_jsonfile, Composition, save_jsonfile

"""
based on vsepr shapes
"""

import json


def cal_bu_class(bu: BuildingUnit):
    if bu.composition == Composition("C O2"):
        return "linear"
    if bu.composition["O"] == 4 and bu.composition["C"] == 0:
        return "tetrahedral-4"
    if bu.composition in [Composition(f) for f in ["B F O3", "As H O3", "P H O3", ]]:
        return "tetrahedral-3"
    if bu.composition in [Composition(f) for f in ["P H2 O2", ]]:
        return "tetrahedral-2"
    if bu.composition in [Composition(f) for f in ["S O3", "Se O3", "As O3"]]:
        return "trigonal pyramidal-3"
    if bu.composition in [Composition(f) for f in ["N O3", "I O3", "C O3", "B O3"]]:
        return "trigonal planar-3"
    if bu.composition == Composition("C2 O4"):
        return "oxylate"
    if bu.composition["O"] == 2 and bu.composition["C"] > 0:
        return "carboxyl"


bucurate = pd.read_csv("../1_ChemicalDiagramSearch/4_bucurate.csv")
curated_buids = []
for bus in bucurate["bus"]:
    curated_buids += json.loads(bus)
curated_buids = sorted(set(curated_buids))
bujsons = read_jsonfile("../1_ChemicalDiagramSearch/3_bulist.json")
bulist = [bu for bu in bujsons if bu.buid in curated_buids]
buclass = {str(bu.buid) + "  " + bu.composition.formula: cal_bu_class(bu) for bu in bulist}
pprint(buclass)
buid2class = {bu.buid: cal_bu_class(bu) for bu in bulist}

save_jsonfile(buid2class, "buid_to_class.json")

buid2count = dict(zip([bu.buid for bu in bulist], [0, ] * len(bulist)))
num_nobu = 0
for bus in bucurate["bus"]:
    buids = json.loads(bus)
    if len(buids) == 0:
        num_nobu += 1
    for k in buid2count.keys():
        if k in buids:
            buid2count[k] += 1

bu2class = dict(zip(bulist, [cal_bu_class(bu) for bu in bulist]))

import matplotlib.pyplot as plt
plt.style.use('ggplot')

bu_class_count = dict()
for bu, buclass in bu2class.items():
    if buclass not in bu_class_count.keys():
        bu_class_count[buclass] = buid2count[bu.buid]
    else:
        bu_class_count[buclass] += buid2count[bu.buid]

xs = ["no BU"]
ys = [num_nobu]
for x, y in bu_class_count.items():
    xs.append(x)
    ys.append(y)

x_pos = [i for i, _ in enumerate(xs)]

plt.bar(x_pos, ys, color='green')
plt.xticks(rotation=60)
plt.title("Building Unit Class")
plt.ylabel("Count")

plt.xticks(x_pos, xs)
plt.tight_layout()
plt.savefig("bustack.png", dpi=300)
plt.clf()
