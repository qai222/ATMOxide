import pandas as pd
import ast

csd_df = pd.read_csv("../../../DataGeneration/5_SimpleInput/input.csv")
new_df = pd.read_csv("1_final_input.csv")

for field in ["elements","bus"]:
    unique = set()
    for v in csd_df[field]:
        l = ast.literal_eval(v)
        unique.add(tuple(sorted(l)))
    unique_new = set()
    for v in new_df[field]:
        l = ast.literal_eval(v)
        unique_new.add(tuple(sorted(l)))
    print(field, len(unique_new) - len(unique.intersection(unique_new)))
    print(field, [v for v in unique_new if v not in unique])

for field in ["smiles"]:
    unique = set()
    for v in csd_df[field]:
        unique.add(v)
    unique_new = set()
    for v in new_df[field]:
        unique_new.add(v)
    print(field, len(unique_new) - len(unique.intersection(unique_new)))
    print(field, [v for v in unique_new if v not in unique])
