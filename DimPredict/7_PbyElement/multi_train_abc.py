import numpy as np
from sklearn.model_selection import KFold

from AnalysisModule.routines.util import load_pkl, yaml_fileread
from MLModule.fnn_trainers import get_data_and_net, fnn_train
from MLModule.metric import npallargmax, npargnonzero, npallargmin
from MLModule.models import Fnn_kargs
from MLModule.utils import SEED


def mindex(y_pred, y_true, true_index, mode):
    if mode == "a":
        true_args_method = npallargmax
        pred_args_method = npallargmax
    elif mode == "b":
        true_args_method = npargnonzero
        pred_args_method = npallargmax
    elif mode == "c":
        true_args_method = npallargmin
        pred_args_method = npallargmin
    else:
        raise NotImplementedError("{} not implemented!".format(mode))

    match_irows = []
    mismatch_irows = []
    task_status = {}
    for i in range(len(y_true)):
        true_args = true_args_method(y_true[i])
        pred_args = pred_args_method(y_pred[i])
        if set(dummy_t for dummy_t in true_args).intersection(set(dummy_p for dummy_p in pred_args)):
            task_status[true_index[i]] = 1
            match_irows.append(true_index[i])
        else:
            task_status[true_index[i]] = 0
            mismatch_irows.append(true_index[i])
    return match_irows, mismatch_irows, task_status


best_params = yaml_fileread("../5_HTune/dataset_eab_aemdes.yml")
for k, v in best_params.items():
    Fnn_kargs[k] = v
Fnn_kargs["verbose"] = 0
DataSet, X_train, X_test, y_train, y_test, net = get_data_and_net("../4_DatasetEncoding/dataset_eab_aemdes.pkl",
                                                                  nn_kargs=Fnn_kargs, split_seed=SEED)
X = DataSet.data
y = DataSet.target
task_cols = {}
for kfseed in range(50):
    kf = KFold(n_splits=5, shuffle=True, random_state=kfseed)
    print("running seed:", kfseed)
    for train_index, test_index in kf.split(DataSet.data, DataSet.target):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trained_net = fnn_train(SEED, net, X_train, y_train, "eab_aemdes", wdir="./", tags=["7_PbyElement", ],
                                x_test=X_test, y_test=y_test, use_neptune=False)
        y_pred = trained_net.predict(X_test)
        for mode in ["a", "b", "c"]:
            task_col_name = "task_{}_{}".format(kfseed, mode)
            if task_col_name not in task_cols.keys():
                task_cols[task_col_name] = {}

            match_irows, mismatch_irows, task_status = mindex(y_pred, y_test, test_index, mode)
            for k, v in task_status.items():
                if k in task_cols[task_col_name].keys():
                    print("Wrong!")
                task_cols[task_col_name][k] = v

RawDataset = load_pkl("../1_Condensation/condensed_elements-smiles-bus.pkl")
all_df_dimension = [npallargmax(v)[0] for v in RawDataset.target]
all_df = RawDataset.data.copy()
all_df["dimension"] = all_df_dimension
for task_col_name, task_col in task_cols.items():
    task_col_values = np.zeros(len(task_col.keys()), dtype=bool)
    for k, v in task_col.items():
        task_col_values[k] = v
    all_df[task_col_name] = task_col_values
all_df.to_csv("all_abc.csv", index=False)
