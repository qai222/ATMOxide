import os
from pathlib import Path
from typing import Union

import neptune
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from skorch import NeuralNet, callbacks

from AnalysisModule.routines.util import load_pkl, save_pkl
from MLModule.metric import DimScoreModes, DimScore
from MLModule.models import FnnModule
from MLModule.utils import set_seed, neptune_api_token, neptune_proj_name


def get_data_and_net(dataset_location: Union[Path, str], nn_kargs: dict, split_seed: int):
    DataSet = load_pkl(dataset_location)
    if isinstance(DataSet.data, pd.DataFrame):
        DataSet.data = DataSet.data.values
    nn_kargs["module__d_in"] = DataSet.data.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(DataSet.data, DataSet.target, test_size=0.2,
                                                        random_state=split_seed)
    net = NeuralNet(FnnModule, **nn_kargs)
    return DataSet, X_train, X_test, y_train, y_test, net


def fnn_train(seed, net: NeuralNet, x_train, y_train, expt_name, wdir, tags: [str], use_neptune=True, x_test=None,
              y_test=None):
    set_seed(seed)
    wdir = os.path.abspath(wdir)

    whereami = os.path.abspath(os.getcwd())
    os.chdir(wdir)

    init_params = net.get_params()
    init_params_pkl = '{}_init_params.pkl'.format(expt_name)
    save_pkl(init_params, init_params_pkl)

    if use_neptune:
        neptune.init(api_token=neptune_api_token, project_qualified_name=neptune_proj_name)
        experiment = neptune.create_experiment(
            name=expt_name,
            tags=tags,
            params=net.get_params(), )
        neptune_logger = callbacks.NeptuneLogger(experiment, close_after_train=False)
        net.callbacks.append(neptune_logger)

    net.fit(x_train, y_train)

    params_pkl = '{}_params.pkl'.format(expt_name)
    history_pkl = '{}_history.json'.format(expt_name)
    net.save_params(
        f_params=params_pkl,
        f_history=history_pkl,
    )

    test_loss = None
    if x_test is not None and y_test is not None:
        y_pred = net.predict(x_test)
        if not torch.cuda.is_available():
            test_loss = net.get_loss(torch.from_numpy(y_pred), torch.from_numpy(y_test))
        else:
            test_loss = net.get_loss(torch.from_numpy(y_pred).cuda(), torch.from_numpy(y_test).cuda())
        for mode in DimScoreModes.keys():
            dims = DimScore(y_test, y_pred, mode)
            if use_neptune:
                neptune_logger.experiment.log_metric("test_dimscore_{}".format(mode), dims)

    if use_neptune:
        neptune_logger.experiment.log_metric("test_loss", test_loss)
        neptune_logger.experiment.log_artifact(params_pkl)
        neptune_logger.experiment.log_artifact(init_params_pkl)
        neptune_logger.experiment.log_artifact(history_pkl)
        neptune_logger.experiment.stop()
    os.chdir(whereami)
    return net


def reload_net(init_nn_kargs_pkl: Union[Path, str], net_params_pkl: Union[Path, str] = None, warm_start=False):
    if isinstance(init_nn_kargs_pkl, str):
        init_nn_kargs = load_pkl(init_nn_kargs_pkl)
    else:
        init_nn_kargs = init_nn_kargs_pkl
    if warm_start:
        init_nn_kargs["warm_start"] = True
    if "module" in init_nn_kargs.keys():
        net = NeuralNet(**init_nn_kargs)
    else:
        net = NeuralNet(FnnModule, **init_nn_kargs)
    if net_params_pkl is None:
        return net
    net.initialize()
    net.load_params(f_params=net_params_pkl)
    return net
