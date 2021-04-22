from AnalysisModule.routines.util import load_pkl
import pandas as pd

ds = load_pkl("dataset_eab_mdes.pkl")
mdes_col_names = [c for c in ds.data.columns[96:]]
print(len(mdes_col_names))
df = pd.DataFrame()
df["encoded_mdes"] = mdes_col_names
df.to_csv("molecular_descriptors.csv")