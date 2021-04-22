import os

from AnalysisModule.calculator.descriptors import MolecularDC
from AnalysisModule.routines.data_settings import SDPATH
from AnalysisModule.routines.util import save_pkl, load_pkl
import pandas as pd

smis = [
    "NC1CCNCC1",
    "CN(C)C(C)(C)CCN",
    "CCCCCCCCN1CCNCC1",
    "NCC1CCC(CN)CC1",
]
mdc = MolecularDC(smis)

df1 = mdc.cal_RdkitFrag(smis)
df2 = mdc.cal_Jchem2D(smis)
df3 = mdc.cal_Mordred2D(smis)
df = pd.concat([df1, df2, df3], axis=1)
original_amine_table = load_pkl("AmineTable.pkl")
original_amine_table_fields = list(list(original_amine_table.values())[0].keys())
df = df[[c for c in df.columns if c in original_amine_table_fields]]
print(len(df.columns))
print(len(original_amine_table_fields))
print(set(original_amine_table_fields).difference(set(df.columns)))
df["smiles"] = smis
mdes_df = df.set_index("smiles")
mdes_df = mdes_df.dropna(axis=1, how='any')
amine2mdes = mdes_df.to_dict("index")
save_pkl(amine2mdes, "extra_amine_table.pkl")
