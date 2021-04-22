import matplotlib.pyplot as plt
from itertools import groupby
from AnalysisModule.routines.util import save_jsonfile, extra_smis
import hdbscan
import numba
from rdkit.Chem import Draw
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from collections import Counter
import pandas as pd

# global var
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha':0.5, 's':80, 'linewidth':0}
seed = 42
count_lower_limit = 0
np.random.seed(seed)
"""
https://iwatobipen.wordpress.com/2018/02/23/chemical-space-visualization-and-clustering-with-hdbscan-and-rdkit-rdkit/
https://github.com/iwatobipen/chemo_info/blob/master/chemicalspace2/HDBSCAN_Chemoinfo.ipynb
"""

def get_smiles2mol(extra=False):
    """load mol and generate morgan fp"""
    df = pd.read_csv("../5_SimpleInput/input.csv")
    smiles_counter = Counter(df["smiles"])
    smiles = sorted(set([smi for smi in smiles_counter.keys() if smiles_counter[smi] >= count_lower_limit]))
    if extra:
        smiles += extra_smis
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    for mol in mols:
        AllChem.Compute2DCoords(mol)
    smiles2mol = dict(zip(smiles, mols))
    X = []
    for mol in mols:
        arr = np.zeros((0,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
    print('{} mols loaded'.format(len(X)))
    return X, smiles, smiles2mol, smiles_counter


@numba.njit()
def tanimoto_dist(a,b):
    dotprod = np.dot(a,b)
    tc = dotprod / (np.sum(a) + np.sum(b) - dotprod)
    return 1.0-tc

def dimreduction(input, method="umap"):
    if method == "tsne":
        transformer = TSNE(n_components=2, metric=tanimoto_dist, random_state=seed)
    elif method == "umap":
        transformer = umap.UMAP(n_neighbors=20, min_dist=0.3, metric=tanimoto_dist, random_state=seed)
    else:
        raise NotImplementedError("unknown method: {}".format(method))
    x = transformer.fit_transform(input)
    return x

def clustering(input):
    hdb = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=10, gen_min_span_tree=False)
    hdb.fit(input)
    return hdb.labels_

def plot2d(x2d, labels=None, savename="2d.png"):
    if labels is None:
        plt.scatter(x2d.T[0], x2d.T[1], color='gray', **plot_kwds)
    else:
        plt.scatter(x2d.T[0], x2d.T[1], c = labels, cmap='plasma', **plot_kwds)

    plt.xticks([])
    plt.yticks([])
    plt.box(on=None)
    plt.savefig(savename)
    plt.clf()


if __name__ == '__main__':
    x, smiles, smiles2mol, smiles_counter = get_smiles2mol(extra=True)
    x2d = dimreduction(x, method="umap")
    labels = clustering(x2d)
    smiles_labeled = dict(zip(smiles, list(labels)))
    print(len(x))
    print(len(set(labels)))
    print(labels)
    save_jsonfile(smiles_labeled, "smiles_labeled_umap_extra.json")

