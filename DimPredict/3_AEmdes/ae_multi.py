import logging
import math
import sys
from copy import deepcopy

import pandas as pd
import torch
from sklearn import preprocessing

from AnalysisModule.routines.util import save_pkl
from MLModule.models import AutoEncoder, ae_kargs, AeNet
from MLModule.utils import SEED, VarianceThreshold, load_input_tables, split_columns, set_seed



def bow_tie_sizes(d_in, multiplier=0.8, n=14):
    bt = []
    bt_size = d_in
    for i in range(n):
        bt_size *= multiplier
        bt.append(math.floor(bt_size))
    return tuple(bt)


if __name__ == '__main__':
    set_seed(SEED)
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    AmineTable, BuidTable, ElementTable, IdentifierTable = load_input_tables()
    amine_data = pd.DataFrame.from_dict(AmineTable, orient="index")
    smis = amine_data.index.tolist()
    numerical, obj = split_columns(amine_data)

    values = preprocessing.MinMaxScaler().fit_transform(numerical)
    data = VarianceThreshold(threshold=1e-5).fit_transform(values)
    ae_kargs["module__d_in"] = data.shape[1]
    ae_kargs["verbose"] = 10
    bowties = dict()

    for n in range(2, 14, 1):
        bow_tie = bow_tie_sizes(data.shape[1], n=n)
        ae_kargs["module__hlayer_sizes"] = bow_tie
        bow_tie_mid = bow_tie[-1]
        X = deepcopy(data)
        X = torch.from_numpy(X).float()
        net = AeNet(AutoEncoder, **ae_kargs)
        bowties[bow_tie_mid] = net
        print("--- training bowtie size:", bow_tie_mid)
        net.fit(X, X)
        decoded, encoded = net.forward(X)

        X = X.to(device="cuda")
        encoded = encoded.to(device="cuda")
        decoded = decoded.to(device="cuda")

        loss = net.get_loss(y_pred=(decoded, encoded), y_true=X).item()
        print("loss:", loss)
        cols = ["encoded_mdes_{}".format(i) for i in range(encoded.shape[1])]

        encoded_df = pd.DataFrame(encoded.cpu().detach().numpy(), columns=cols)
        encoded_df["smiles"] = smis
        encoded_df = encoded_df.set_index("smiles")
        encoded_df.to_csv("./progression/mdes_bowtie_{}.csv".format(bow_tie_mid))
    save_pkl(bowties, "./progression/bowties.pkl")
