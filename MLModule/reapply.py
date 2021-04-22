from pathlib import Path
from typing import Union

from sklearn.model_selection import train_test_split

from AnalysisModule.routines.util import load_pkl
from MLModule.fnn_trainers import fnn_train
from MLModule.fnn_trainers import reload_net
from MLModule.utils import SEED


def reapply_net(old_initp: dict, old_p: Union[Path, str], dataset_path: Union[Path, str], splitseed, oldnetname,
                newnetname):
    dataset = load_pkl(dataset_path)
    old_initp["module__d_in"] = dataset.data.shape[1]
    new_net = reload_net(old_initp, old_p, warm_start=True)
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2,
                                                        random_state=splitseed)
    trained_net = fnn_train(SEED, new_net, X_train.values, y_train, "apply_{}_to_{}".format(oldnetname, newnetname),
                            wdir="./",
                            tags=["transfer", "2080"],
                            x_test=X_test.values, y_test=y_test, use_neptune=False)
    return trained_net


def reapply_net_nosplit(old_initp: dict, old_p: Union[Path, str], X_train, y_train, X_test, y_test, oldnetname,
                        newnetname):
    old_initp["module__d_in"] = X_train.shape[1]
    new_net = reload_net(old_initp, old_p, warm_start=True)
    trained_net = fnn_train(SEED, new_net, X_train, y_train, "apply_{}_to_{}".format(oldnetname, newnetname), wdir="./",
                            tags=["transfer", "2080"],
                            x_test=X_test, y_test=y_test, use_neptune=False)
    return trained_net
