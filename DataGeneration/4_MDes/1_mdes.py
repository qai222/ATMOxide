import os

from AnalysisModule.calculator.descriptors import MolecularDC
from AnalysisModule.routines.data_settings import SDPATH
import pandas as pd

thisdir = os.path.dirname(os.path.abspath(__file__))
mdes_folder = SDPATH.mdes_data
df_amine = pd.read_csv("{}/../1_ChemicalDiagramSearch/4_bucurate.csv".format(thisdir))

identifierandsmiles = list(zip(df_amine["identifier"], df_amine["smiles"]))
smis = df_amine["smiles"].tolist()
mdc = MolecularDC(smis)

df = mdc.cal_RdkitFrag(smis)
df.to_csv("1_mdes_rdkit.csv", index=False)
print("rdkit fin")
df = mdc.cal_Jchem2D(smis)
df.to_csv("1_mdes_jchem.csv", index=False)
print("jchem fin")

from tqdm import tqdm
results = []
import numpy as np
np.seterr(all = 'raise')
for identifier, smi in tqdm(identifierandsmiles):
    r = dict(identifier=identifier)
    try:
        nr = mdc.cal_Mordred2D(smi)
        for k, v in nr.items():
            r[k] = v
    except FloatingPointError:
        print(identifier, smi)
    results.append(r)
df = pd.DataFrame.from_records(results)
df.to_csv("1_mdes_mordred.csv", index=False)
print("mordred fin")

df1 = pd.read_csv("1_mdes_mordred.csv")
df2 = pd.read_csv("1_mdes_jchem.csv")
df3 = pd.read_csv("1_mdes_rdkit.csv")

df = pd.concat([df1, df2, df3], axis=1)
df.to_csv("1_mdes.csv", index=False)
