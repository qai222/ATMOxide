from MLModule.fnn_trainers import get_data_and_net
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
from MLModule.fnn_trainers import reload_net
from MLModule.metric import DimScoreModes, DimScore, npallargmax, npargnonzero, npallargmin
from MLModule.models import Fnn_kargs
from MLModule.utils import SEED


def task_output(y_true, y_pred, task):
    dim_pred = []
    dim_true = []
    assert y_true.shape == y_pred.shape
    if task == "a":
        true_args_method = npallargmax
        pred_args_method = npallargmax
    elif task == "b":
        true_args_method = npargnonzero
        pred_args_method = npallargmax
    elif task == "c":
        true_args_method = npallargmin
        pred_args_method = npallargmin
    else:
        raise NotImplementedError("{} not implemented!".format(task))
    for i in range(len(y_true)):
        true_args = true_args_method(y_true[i])
        pred_args = pred_args_method(y_pred[i])
        if set(pred_args).intersection(set(true_args)):
            dim_pred.append(pred_args[0])
            dim_true.append(pred_args[0])
        else:
            dim_pred.append(pred_args[0])
            dim_true.append(true_args[0])
    return dim_pred, dim_true

def get_cm_dfs(dim_pred, dim_true):
    cm_ntrue = confusion_matrix(dim_true, dim_pred, normalize='true')
    cm_npred = confusion_matrix(dim_true, dim_pred, normalize='pred')

    df_ntrue = pd.DataFrame(cm_ntrue, index=["true {}".format(i) for i in "0123"],
                         columns=["predict {}".format(i) for i in "0123"])
    df_npred = pd.DataFrame(cm_npred, index=["true {}".format(i) for i in "0123"],
                         columns=["predict {}".format(i) for i in "0123"])
    return df_ntrue, df_npred

def get_cm_dfs_nonorm(dim_pred, dim_true):
    cm = confusion_matrix(dim_true, dim_pred, normalize=None)
    df = pd.DataFrame(cm, index=["true {}".format(i) for i in "0123"],
                            columns=["predict {}".format(i) for i in "0123"])
    return df

def heat_triangles(df_true, df_pred, taskname):
    mat = np.eye(4)
    mat[3, 0] = 1

    M = 4
    N = 4
    x = np.arange(M + 1)
    y = np.arange(N + 1)
    xs, ys = np.meshgrid(x, y)
    zs_pred = np.zeros((len(xs), len(ys)))
    for i in range(zs_pred.shape[0] - 1):
        for j in range(zs_pred.shape[1] - 1):
            zs_pred[i, j] = df_pred.values[i, j]
    zs_pred = zs_pred[:-1, :-1].ravel()
    zs_true = np.zeros((len(xs), len(ys)))
    for i in range(zs_true.shape[0] - 1):
        for j in range(zs_true.shape[1] - 1):
            zs_true[i, j] = df_true.values[i, j]
    zs_true = zs_true[:-1, :-1].ravel()

    triangles1 = [
        (
            i + j * (M + 1),  # 0 0
            i + 1 + (j + 1) * (M + 1),  # 1 1
            i + (j + 1) * (M + 1)  # 0 1
        ) for j in range(N) for i in range(M)]
    triangles2 = [
        (
            i + j * (M + 1),  # 0 0
            i + 1 + j * (M + 1),  # 1 0
            i + 1 + (j + 1) * (M + 1),  # 1 1
        ) for j in range(N) for i in range(M)]
    triang1 = Triangulation(xs.ravel(), ys.ravel(), triangles1)  # top left
    triang2 = Triangulation(xs.ravel(), ys.ravel(), triangles2)  # bot right
    img1 = plt.tripcolor(triang1, zs_true, cmap=plt.get_cmap('copper', 100), vmax=1, vmin=0)
    img2 = plt.tripcolor(triang2, zs_pred, cmap=plt.get_cmap('copper', 100), vmax=1, vmin=0)
    plt.colorbar(img1, ticks=range(10))
    plt.xlim(x[0], x[-1])
    plt.ylim(y[0], y[-1])
    plt.xticks([i + 0.5 for i in range(4)], df_pred.columns)
    plt.yticks([i + 0.5 for i in range(4)], ["true {}".format(i) for i in range(4)])
    for i in range(4):
        for j in range(4):
            c_x = i + 0.5
            c_y = j + 0.5
            top_left_center = (c_x - 0.35, c_y + 0.25)
            bot_right_center = (c_x, c_y - 0.25)
            plt.text(top_left_center[0], top_left_center[1], "{:.1%}".format(df_true.values[j, i]), c="white")
            plt.text(bot_right_center[0], bot_right_center[1], "{:.1%}".format(df_pred.values[j, i]), c="white")
    plt.savefig("cm_{}.png".format(taskname), dpi=300)
    plt.savefig("cm_{}.eps".format(taskname))
    plt.clf()

if __name__ == '__main__':
    DataSet, X_train, X_test, y_train, y_test, net = get_data_and_net(
        "../../DimPredict/4_DatasetEncoding/dataset_eab_aemdes.pkl",
        nn_kargs=Fnn_kargs, split_seed=SEED)
    net = reload_net("../../DimPredict/6_TrainOpt/dataset_eab_aemdes_init_params.pkl",
                     "../../DimPredict/6_TrainOpt/dataset_eab_aemdes_params.pkl")
    y_pred = net.predict(X_test)
    for a in DimScoreModes:
        print(a, DimScore(y_test, y_pred, a))
        sn.set(font_scale=1.8)
        dim_pred, dim_true = task_output(y_test, y_pred, a)
        df_ntrue, df_npred = get_cm_dfs(dim_pred, dim_true)
        df_nonorm = get_cm_dfs_nonorm(dim_pred, dim_true)

        plt.figure(figsize=(10, 7))
        sn.heatmap(df_nonorm, annot=True, fmt="g", vmin=0, vmax=250)
        plt.title("Task {}".format(a.upper()))
        plt.savefig("traditional_{}.eps".format(a))
        plt.clf()

        sn.set(font_scale=1)
        heat_triangles(df_ntrue, df_npred, a)


