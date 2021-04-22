import numpy as np
import torch
from sklearn.metrics import make_scorer

from MLModule.loss import CELoss


def tensor2np(t: torch.Tensor):
    return t.cpu().detach().numpy()


def np2tensor(a: np.ndarray):
    return torch.from_numpy(a)


def npallargmax(a):
    """
    basically np.argmax that returns all inds, works ok for 1d array
    """
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return all_


def npallargmin(a):
    """
    basically np.argmin that returns all inds, works ok for 1d array
    """
    if len(a) == 0:
        return []
    all_ = [0]
    min_ = a[0]
    for i in range(1, len(a)):
        if a[i] < min_:
            all_ = [i]
            min_ = a[i]
        elif a[i] == min_:
            all_.append(i)
    return all_


def npargnonzero(a):
    if a.ndim == 1:
        return np.nonzero(a)[0]
    else:
        return np.nonzero(a)


def npfirstNargmax(a, N=-2):
    return np.argpartition(a, N)[N:]


DimScoreModes = dict(
    a="the most probable dimension predicted matches the most probable dimension in y_true",
    b="the most probable dimension predicted has a non-zero probability in y_true",
    c="the most unlikely dimension predicted matches the most unlikely dimension in y_true"
)


def DimScore(y_true, y_pred, mode="a"):
    """
    score function for dimension probability vectors

    :param y_true: Mx4 array
    :param y_pred: Mx4 array
    :param mode:
        a  = "the most probable dimension predicted matches the most probable dimension in y_true",
        b  = "the most probable dimension predicted has a non-zero probability in y_true",
        c  = "the most unlikely dimension predicted matches the most unlikely dimension in y_true"
    :return:
    """
    assert y_true.shape == y_pred.shape
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

    imatch = 0
    for i in range(len(y_true)):
        true_args = true_args_method(y_true[i])
        pred_args = pred_args_method(y_pred[i])
        if set(dummy_t for dummy_t in true_args).intersection(set(dummy_p for dummy_p in pred_args)):
            imatch += 1
    return imatch / len(y_true)


def np_celoss(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = tensor2np(y_true)
    if torch.is_tensor(y_pred):
        y_pred = tensor2np(y_pred)
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    cearray = np.zeros(y_pred.shape[0])
    for i, invector in enumerate(y_pred):
        cearray[i] = - sum(np.multiply(y_true[i], np.log(y_pred[i])))
    return np.mean(cearray)


def np_celoss_pair(y1, y2):
    y1 = np.clip(y1, 1e-9, 1 - 1e-9)
    y2 = np.clip(y2, 1e-9, 1 - 1e-9)
    return - sum(np.multiply(y1, np.log(y2)))


def torch_celoss(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        y_true = np2tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = np2tensor(y_pred)
    return CELoss().forward(y_pred=y_pred, y_true=y_true).item()


CELossScorer = make_scorer(torch_celoss, greater_is_better=False)
# less negative meaning better

DimScorers = {}
for m in DimScoreModes.keys():
    DimScorers[m] = make_scorer(DimScore, greater_is_better=True, mode=m)
