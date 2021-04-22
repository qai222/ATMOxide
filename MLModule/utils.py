import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold

from AnalysisModule.routines.util import load_pkl

this_dir = os.path.dirname(os.path.abspath(__file__))

SEED = 42


def load_input_tables():
    InputTables = dict()
    for table_pkl in glob.glob("{}/../DataGeneration/9_LookupTables/*.pkl".format(this_dir)):
        name = os.path.basename(table_pkl).replace(".pkl", "")
        InputTables[name] = load_pkl(table_pkl)
    AmineTable = InputTables['AmineTable']
    BuidTable = InputTables['BuidTable']
    ElementTable = InputTables['ElementTable']
    IdentifierTable = InputTables['IdentifierTable']
    return AmineTable, BuidTable, ElementTable, IdentifierTable


def split_columns(df: pd.DataFrame):
    objcolumns = []
    numerical_columns = []
    for c in df.columns:
        if df[c].dtypes in (np.dtype(np.int64), np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.float32)):
            numerical_columns.append(c)
        else:
            objcolumns.append(c)
        # elif dataset.data[c].dtypes in (np.dtype(object),):
        #     objcolumns.append(c)
    return df[numerical_columns], df[objcolumns]


def variance_threshold_selector(data, threshold=0.2):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


# enter your own api token for neptune here
neptune_api_token = None
neptune_proj_name = None
