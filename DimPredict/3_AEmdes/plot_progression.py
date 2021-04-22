import matplotlib.pylab as plt

plt.rcParams.update({'font.size': 16})
from copy import deepcopy

import torch
from MLModule.utils import VarianceThreshold, load_input_tables, split_columns
from AnalysisModule.routines.util import load_pkl
from sklearn import preprocessing

if __name__ == '__main__':
    import pandas as pd

    AmineTable, BuidTable, ElementTable, IdentifierTable = load_input_tables()
    amine_data = pd.DataFrame.from_dict(AmineTable, orient="index")
    numerical, obj = split_columns(amine_data)
    values = preprocessing.MinMaxScaler().fit_transform(numerical)
    data = VarianceThreshold(threshold=1e-5).fit_transform(values)

    bowties = load_pkl("./progression/bowties.pkl")
    xs = []
    ys = []
    for bow_tie_mid, net in bowties.items():
        X = deepcopy(data)
        X = torch.from_numpy(X).float()
        decoded, encoded = net.forward(X)

        X = X.to(device="cuda")
        encoded = encoded.to(device="cuda")
        decoded = decoded.to(device="cuda")

        loss = net.get_loss(y_pred=(decoded, encoded), y_true=X).item()
        xs.append(bow_tie_mid)
        ys.append(float(loss))

    print(xs, ys)
    plt.plot(xs, ys, "ro:")
    plt.xlabel("Output dimension")
    plt.ylabel("MSE loss")
    plt.xlim((max(xs)+10, min(xs)-10))
    plt.tight_layout()
    plt.savefig("progression.png")
    plt.savefig("progression.eps")
