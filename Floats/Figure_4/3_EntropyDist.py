from MLModule.metric import np_celoss_pair
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import entr
sns.set_style("white")
sns.set(font_scale=2)
kwargs = dict(hist_kws={'alpha': .4}, kde_kws={'linewidth': 2, "bw_adjust": 0.3})

df_by_ele = pd.read_csv("2_CountDF_EP.csv")
df_by_ami = pd.read_csv("2_CountDF_TA.csv")


def get_p_mat(df):
    p = df[["0", "1", "2", "3"]].values
    p = p.astype(float)
    for i in range(len(p)):
        p[i] = p[i] / sum(p[i])
    return p

def get_baseline(df):
    p = df[["0", "1", "2", "3"]].values
    p = p.astype(float)
    b = np.sum(p, axis=0)
    return b / sum(b)

def get_celoss_list(df):
    p = get_p_mat(df)
    b = get_baseline(df)
    ce = []
    for i in range(len(p)):
        celoss = np_celoss_pair(b, p[i])
        ce.append(np.log(celoss))
    return ce

def get_entropy(df):
    p = get_p_mat(df)
    e = []
    for i in range(len(p)):
        e.append(entr(p[i]).sum())
    return e

def plot_e_dist():
    ce_ele = get_entropy(df_by_ele)
    ce_ami= get_entropy(df_by_ami)
    bins = np.linspace(0, 1.5, 20)

    plt.clf()
    plt.figure(figsize=(10,7))
    sns.distplot(ce_ele, bins=bins, color="red", label=r"$P^{\mathrm{elements}}$", **kwargs)
    sns.distplot(ce_ami, bins=bins, color="green", label=r"$P^{\mathrm{amine}}$", **kwargs)
    plt.xlim(-0.01,1.5)
    plt.legend()
    plt.xlabel("Entropy")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("3_EntropyDist.png", dpi=300)
    plt.savefig("3_EntropyDist.pdf")

def plot_ce_dist():
    ce_ele = get_celoss_list(df_by_ele)
    ce_ami= get_celoss_list(df_by_ami)
    bins = np.linspace(0, 3, 30)

    plt.clf()
    plt.figure(figsize=(10, 7))
    sns.distplot(ce_ele, bins=bins, color="red", label=r"$P^{\mathrm{elements}}$", **kwargs)
    sns.distplot(ce_ami, bins=bins, color="green", label=r"$P^{\mathrm{amine}}$", **kwargs)
    plt.xlabel("Log cross entropy against baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("3_CEntropyDist.png", dpi=300)
    plt.savefig("3_CEntropyDist.pdf")

if __name__ == '__main__':
    plot_e_dist()
    plot_ce_dist()

