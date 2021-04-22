from AnalysisModule.routines.util import load_pkl
import os
import glob
from MLModule.metric import DimScoreModes, DimScore, np, np_celoss
from MLModule.utils import SEED
from sklearn.model_selection import train_test_split


def cal_baseline(condensed_dataset, split=True):
    if split:
        X_train, X_test, y_train, y_test = train_test_split(condensed_dataset.data, condensed_dataset.target, test_size=0.2,
                                                            random_state=SEED)
        target = y_test
    else:
        target = condensed_dataset.target

    baselines = dict()
    for mode in DimScoreModes.keys():
        baseline = 0
        baseline_pred = None
        for i in [0, 1, 2, 3]:
            if mode in ["a", "b"]:
                y_pred = np.zeros((len(target), 4))
                y_pred[:, i] = 1
            elif mode == "c":
                y_pred = np.ones((len(target), 4))
                y_pred[:, i] = 0
            dims = DimScore(target, y_pred, mode)
            if dims > baseline:
                baseline = dims
                baseline_pred = i
        baselines[mode] = ("pred: {}".format(baseline_pred), baseline)
    return baselines

if __name__ == '__main__':
    import sys
    sys.stdout = open("tasks.log", "w")
    for pkl in glob.glob("../1_Condensation/condensed_*.pkl"):
        ds = load_pkl(pkl)
        dsname = os.path.basename(pkl).replace(".pkl", "")
        baseline = cal_baseline(ds, split=True)
        print("---", dsname, ":")
        print(baseline)
