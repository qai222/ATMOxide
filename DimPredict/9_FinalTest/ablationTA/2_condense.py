import pandas as pd
from MLModule.condenser import Condenser
from AnalysisModule.routines.util import load_pkl, save_pkl
import numpy as np

IF_list = [
    ["elements", "smiles", "bus"],
]
INPUT_DF = pd.read_csv("1_final_input.csv")
total_df = pd.read_csv("../../../DataGeneration/5_SimpleInput/input.csv")
total_df = total_df[[c for c in INPUT_DF.columns if c in total_df.columns]]
# print(total_df.shape)
# print(INPUT_DF.shape)
# INPUT_DF = pd.concat([total_df, INPUT_DF], ignore_index=True)
# print(INPUT_DF.shape)
for IF in IF_list:
    ds = "dataset_{}.pkl".format("-".join(IF))
    total_ds = "dataset_total_{}.pkl".format("-".join(IF))
    condenser = Condenser(INPUT_DF, IF)
    condenser.condense_to_dataset(ds)

    condenser_total = Condenser(total_df, IF)
    condenser_total.condense_to_dataset(total_ds)

    ds = load_pkl(ds)
    total_ds = load_pkl(total_ds)
    ds.data = pd.concat([total_ds.data, ds.data], ignore_index=True)
    ds.target = np.concatenate((total_ds.target, ds.target), axis=0)
    save_pkl(ds, "condensed.pkl")