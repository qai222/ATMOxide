import pandas as pd
from AnalysisModule.routines.util import read_jsonfile, save_jsonfile, save_pkl
from pprint import pprint
from AnalysisModule.prepare.diagram import BuildingUnit
import json
import ast
"""
generate lookup dict targeting individual phase_one features
    - identifier
        - year
        - smiles
        - bus
        - ms
        - nms
    - smiles
        - cluster label
        - * mol des
    - buid_x
        - # of oxygen
        - vsepr class
    - m_x
        - * element des
    - nm_x
        - * element des
    
"""
PhaseOne_df = pd.read_csv("../5_SimpleInput/input.csv")


def Get_IdentifierTable():
    # identifier 2 year
    identifier2year = read_jsonfile("../8_PubYear/identifier_labeled_year.json")

    # identifier dict
    identifier_table = dict()
    for r in PhaseOne_df.to_dict("records"):
        identifier = r["identifier"]
        for field in ["bus", "elements", "ms", "nms"]:
            r[field] = ast.literal_eval(r[field])
        r["year"] = identifier2year[identifier]
        identifier_table[identifier] = r
    return identifier_table


def Get_ElementTable():
    # element 2 edes
    # from https://github.com/mtdg-wagner/Elemental-descriptors
    edes_df = pd.read_csv("elements.csv", index_col="Symbol")
    edes_df = edes_df.dropna(axis=1,how='any')
    element2edes = edes_df.to_dict("index")
    return element2edes


def Get_AmineTable():
    # amine 2 cluster label
    amine2clusterlabel = read_jsonfile("../6_AmineCluster/smiles_labeled_umap.json")
    #
    # amine 2 mdes
    mdes_df = pd.read_csv("../4_MDes/1_mdes.csv")
    mdes_df = mdes_df[mdes_df["identifier"].isin(PhaseOne_df["identifier"].tolist())]
    mdes_df = mdes_df.dropna(axis=1,how='any')

    identifier2smiles = pd.Series(PhaseOne_df.smiles.values, index=PhaseOne_df.identifier).to_dict()
    mdes_df["smiles"] = [identifier2smiles[i] for i in mdes_df["identifier"]]


    mdes_df = mdes_df.drop_duplicates(subset=["smiles"])
    mdes_df = mdes_df.set_index("smiles")
    amine2mdes = mdes_df.to_dict("index")
    for smi in amine2mdes.keys():
        amine2mdes[smi]["saoto_clusterlabel"] = amine2clusterlabel[smi]
    return amine2mdes

def Get_BuidTable():
    bulist = read_jsonfile("../1_ChemicalDiagramSearch/3_bulist.json")
    buid2class = read_jsonfile("../7_ClassifyBU/buid_to_class.json")
    buid2class = {int(k):v for k, v in buid2class.items()}
    butable = dict()
    for k in buid2class.keys():
        this_bu = [bu for bu in bulist if bu.buid==k][0]
        this_bu: BuildingUnit
        # butable[k] = this_bu.as_dict()  # too much info...
        butable[k] = dict()
        butable[k]["buclass"] = buid2class[k]
        butable[k]["composition"] = this_bu.composition.formula
        butable[k]["num_oxygens"] = int(this_bu.composition["O"])
    return butable

if __name__ == '__main__':
    import inspect
    import sys
    table_functions = {name:obj for name, obj in inspect.getmembers(sys.modules[__name__]) if (inspect.isfunction(obj)) and name.startswith("Get_")}
    for name, table_func in table_functions.items():
        pklname = name.replace("Get_", "")
        save_pkl(table_func(), "{}.pkl".format(pklname))
