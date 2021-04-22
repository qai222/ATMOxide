import glob
import os
from copy import deepcopy

from AnalysisModule.routines.util import yaml_fileread
from MLModule.fnn_trainers import get_data_and_net, fnn_train
from MLModule.models import Fnn_kargs
from MLModule.utils import SEED

for yml in glob.glob("../5_HTune/*.yml"):
    dataset_name = os.path.basename(yml)[:-4]
    dspkl = "../4_DatasetEncoding/{}.pkl".format(dataset_name)
    best_params = yaml_fileread(yml)
    nnkargs = deepcopy(Fnn_kargs)
    for k, v in best_params.items():
        nnkargs[k] = v
    DataSet, X_train, X_test, y_train, y_test, net = get_data_and_net(dspkl, nn_kargs=nnkargs, split_seed=SEED)
    trained_net = fnn_train(SEED, net, X_train, y_train, dataset_name, wdir="./", tags=["tuned", dataset_name, "2080"],
                            x_test=X_test, y_test=y_test)
