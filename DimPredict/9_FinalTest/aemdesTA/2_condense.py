import pandas as pd
from MLModule.condenser import Condenser
from AnalysisModule.routines.util import load_pkl, save_pkl
import numpy as np

IF_list = [
    ["elements", "smiles", "bus"],
]
INPUT_DF = pd.read_csv("1_final_input.csv")
original_df = pd.read_csv("../../../DataGeneration/5_SimpleInput/input.csv")
original_df = original_df[[c for c in INPUT_DF.columns if c in original_df.columns]]
for IF in IF_list:
    condenser = Condenser(INPUT_DF, IF)
    final_test_ds = condenser.condense_to_dataset()

    condenser_original = Condenser(original_df, IF)
    original_ds = condenser_original.condense_to_dataset()

    total_data = pd.concat([original_ds.data, final_test_ds.data], ignore_index=True)
    total_target = np.concatenate((original_ds.target, final_test_ds.target), axis=0)
    final_test_ds.data = total_data
    final_test_ds.target = total_target
    save_pkl(final_test_ds, "condensed.pkl")