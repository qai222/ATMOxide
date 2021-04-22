import sys
from copy import deepcopy
from MLModule.fnn_trainers import reload_net
from skopt import load, BayesSearchCV
from skorch import NeuralNet
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
import os
from sklearn.model_selection import train_test_split

from MLModule.metric import DimScoreModes, DimScore
from MLModule.fnn_trainers import fnn_train, get_data_and_net, FnnModule
from MLModule.models import Fnn_kargs
from MLModule.utils import SEED, set_seed
from AnalysisModule.routines.util import load_pkl
from MLModule.reapply import reapply_net, reapply_net_nosplit

sys.stdout = open("exace_transfer.log", "w")


def transfer_entries(x_train, less_condensed_set_data):
    """
    find a subset of `less_condensed_set_data` in which all entries appear at least once in x_train
    """
    i2j = dict()
    js = []
    distmat = cdist(x_train, less_condensed_set_data)
    for i in range(distmat.shape[0]):
        match_js = np.where(distmat[i] < 1e-5)[0].tolist()
        i2j[i] = match_js
        js += match_js
    return i2j, sorted(set(js))


def sync_split(ds1, ds2, split_seed):
    if isinstance(ds1.data, pd.DataFrame):
        ds1.data = ds1.data.values
    if isinstance(ds2.data, pd.DataFrame):
        ds2.data = ds2.data.values
    x_train, x_test, y_train, y_test = train_test_split(ds1.data, ds1.target, test_size=0.2, random_state=split_seed)
    i2jdict, jidx = transfer_entries(x_train, ds2.data)
    x_train2 = ds2.data[jidx]
    y_train2 = ds2.target[jidx]
    return x_train, x_test, y_train, y_test, x_train2, y_train2


def print_net_score(net, net_name, x_test, y_test):
    y_pred = net.predict(x_test)
    print("********", net_name)
    for mode in DimScoreModes.keys():
        dims = DimScore(y_test, y_pred, mode)
        print(mode + ":", dims)
    y_pred = torch.from_numpy(y_pred).to(device="cuda")
    y_test = torch.from_numpy(y_test).to(device="cuda")
    test_loss = net.get_loss(y_pred, y_test).item()
    print("test loss:", test_loss)

def get_warm_net(module, netparams_pkl, initargs):
    initargs["warm_start"] = True
    new_net = NeuralNet(module=module, **initargs)
    new_net.initialize()
    new_net.load_params(f_params=netparams_pkl)
    return new_net

def retrain_net(
        seed, warm_net, x_train, y_train,
        expt_name, wdir="./", save=True
):
    set_seed(seed)
    whereami = os.path.abspath(os.getcwd())
    wdir = os.path.abspath(wdir)
    os.chdir(wdir)
    warm_net.fit(x_train, y_train)

    if save:
        params_pkl = '{}_params.pkl'.format(expt_name)
        history_pkl = '{}_history.json'.format(expt_name)
        warm_net.save_params(
            f_params=params_pkl,
            f_history=history_pkl,
        )
    os.chdir(whereami)
    return warm_net


if __name__ == '__main__':
    set_seed(SEED)
    eb_dataset = load_pkl("../4_DatasetEncoding/dataset_eb_ablation.pkl")
    none_dataset = load_pkl("../4_DatasetEncoding/dataset_eab_None.pkl")
    x_train, x_test, y_train, y_test, x_train2, y_train2 = sync_split(eb_dataset, none_dataset, split_seed=SEED)

    # train eb_ablation net using x_train2
    opt = load("../5_HTune/htune_dataset_eb_ablation.pkl")
    opt: BayesSearchCV
    nnkargs = deepcopy(Fnn_kargs)
    for k, v in opt.best_params_.items():
        nnkargs[k] = v
    nnkargs["module__d_in"] = x_train.shape[1]
    nnkargs["verbose"] = 0
    native_net = NeuralNet(FnnModule, **nnkargs)
    native_net.fit(x_train, y_train)

    # train eba_None using x_train2
    opt = load("../5_HTune/htune_dataset_eab_None.pkl")
    opt: BayesSearchCV
    nnkargs = deepcopy(Fnn_kargs)
    for k, v in opt.best_params_.items():
        nnkargs[k] = v
    nnkargs["module__d_in"] = x_train.shape[1]
    nnkargs["verbose"] = 0
    transfer_net = NeuralNet(FnnModule, **nnkargs)
    transfer_net.fit(x_train2, y_train2)
    transfer_net_params_pkl = 'transfer_params.pkl'
    transfer_net.save_params(transfer_net_params_pkl)

    # we can use eba_None as the starting point
    train_transfer_net = reapply_net_nosplit(nnkargs, "transfer_params.pkl", x_train, y_train, x_test, y_test, "transfer", "native")

    print_net_score(native_net, "native_net", x_test, y_test)
    print_net_score(transfer_net, "transfer_net", x_test, y_test)
    print_net_score(train_transfer_net, "train_transfer_net", x_test, y_test)
