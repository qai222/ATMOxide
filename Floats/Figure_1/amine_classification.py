from collections import Counter

import pandas as pd

from AnalysisModule.calculator.descriptors import MolecularDC

smiles = pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")["smiles"]
smiles = sorted(set(smiles))
RdkitFrags = """_feat_fr_NH2 Number of primary amines
_feat_fr_NH1 Number of secondary amines
_feat_fr_NH0 Number of tertiary amines
_feat_fr_quatN Number of quaternary amines
_feat_fr_ArN Number of N functional groups attached to aromatics
_feat_fr_Ar_NH Number of aromatic amines
_feat_fr_Imine Number of imines
_feat_fr_amidine Number of amidine groups
_feat_fr_dihydropyridine Number of dihydropyridines
_feat_fr_guanido Number of guanidine groups
_feat_fr_piperdine Number of piperidine rings
_feat_fr_piperzine Number of piperzine rings
_feat_fr_pyridine Number of pyridine rings"""
mdc = MolecularDC(smiles)
frags = mdc.cal_RdkitFrag(smiles)
selected_cols = [
    "fr_NH2",
    "fr_NH1",
    "fr_NH0"
]
frags = frags[selected_cols]

amine_counter = dict()
amine_counter["fr_NH2"] = 0
amine_counter["fr_NH1"] = 0
amine_counter["fr_NH0"] = 0
amine_counter["mixed"] = 0

for r in frags.to_dict("records"):
    nz_k = []
    for k, v in r.items():
        if v > 0:
            nz_k.append(k)
    if len(nz_k) == 1:
        amine_counter[nz_k[0]] += 1
    else:
        amine_counter["mixed"] += 1
from pprint import pprint

pprint(amine_counter)

label_dict = {
    'fr_NH0': r"3$^\circ$",
    'fr_NH1': r"2$^\circ$",
    'fr_NH2': r"1$^\circ$",
    'mixed': "Mixed",
}

labels = []
sizes = []
for k, v in amine_counter.items():
    labels.append(label_dict[k])
    sizes.append(v)
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0, 0, 0.1)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=False, startangle=90, textprops={'fontsize': 20})
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.title("Total: {}".format(len(smiles)))
plt.axis('equal')
plt.savefig("amine_classification.png", dpi=600)
plt.savefig("amine_classification.eps", dpi=600)
plt.clf()

num_Ns = []
for smi in smiles:
    num_N = Counter(smi.lower())["n"]
    num_Ns.append(num_N)
amine_num_N_counter = Counter(num_Ns)
ngt = 5
num_N_gt = sum(amine_num_N_counter[k] for k in amine_num_N_counter.keys() if k >= ngt)
labels = [1, 2, 3, 4]
sizes = [amine_num_N_counter[l] for l in labels]
labels.append(">={}".format(ngt))
sizes.append(num_N_gt)
# colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'gray']
explode = (0, 0, 0, 0, 0.1)

pie = plt.pie(sizes,
              explode=explode,
              labels=labels,
              # colors=colors,
              autopct='%1.1f%%', shadow=False, startangle=90,textprops={'fontsize': 18}
              )
# Set aspect ratio to be equal so that pie is drawn as a circle.
# plt.legend(pie[0],labels, bbox_to_anchor=(1,0), loc="lower right",
#                           bbox_transform=plt.gcf().transFigure)
plt.title("Total: {}".format(len(smiles)))
plt.axis('equal')
plt.tight_layout()
plt.savefig("amine_classification_numN.png", dpi=600)
plt.savefig("amine_classification_numN.eps", dpi=600)
