import ast
import inspect
import logging
import os
import pathlib
import sys
import typing

import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import _pt_data, Element

from AnalysisModule.calculator.fp import GetFpArray
from AnalysisModule.routines.util import MDefined, NMDefined
from AnalysisModule.routines.util import load_pkl, save_pkl
from MLModule.utils import load_input_tables
from MLModule.utils import variance_threshold_selector, split_columns

this_dir = os.path.dirname(os.path.abspath(__file__))


def Encode_bus(bus_column: [str], BuidTable=None):
    BUid_in_dataset = sorted(BuidTable.keys())
    num_bus = len(BUid_in_dataset)
    buid_encoding_dict = {v: k for k, v in enumerate(BUid_in_dataset)}

    ohe_array = np.zeros((len(bus_column), num_bus), dtype=np.float32)
    for i_entry, bu_entry in enumerate(bus_column):
        for buid in ast.literal_eval(bu_entry):
            ohe_array[i_entry][buid_encoding_dict[buid]] = 1

    logging.info("{} gives # of columns: {}".format(inspect.stack()[0][3], ohe_array.shape[1]))
    columns = ["Buid_bit_{}".format(num) for num in range(ohe_array.shape[1])]
    ohe_df = pd.DataFrame(ohe_array, columns=columns)
    return ohe_array, ohe_df


def Encode_elements(
        compositions: [str],
        possible_elements: [str] = None,
        exclude_elements=("H", "O"),
        exclude_groups=("noble_gas",),
        feature_header="Elements_bit"
):
    """
    one hot encoder for elementary strings
    """
    if possible_elements is None:
        possible_elements = sorted(_pt_data.keys())
    elements = []
    for e in possible_elements:
        if e in exclude_elements:
            continue
        element = Element(e)
        if any(getattr(element, "is_{}".format(p)) for p in exclude_groups):
            continue
        elements.append(e)
    elements = sorted(elements)
    n_compositions = len(compositions)

    ohe_array = np.zeros((n_compositions, len(elements)), dtype=np.float32)

    presented = []
    for icomp, composition in enumerate(compositions):
        symbol_list = ast.literal_eval(composition)
        for string in symbol_list:
            presented.append(string)
            if string == "O":
                continue
            ind = elements.index(string)
            ohe_array[icomp][ind] = 1

    logging.info("{} gives # of columns: {}".format(inspect.stack()[0][3], ohe_array.shape[1]))
    # columns = ["{}_{}".format(feature_header, num) for num in range(ohe_array.shape[1])]
    columns = [elements[num] for num in range(ohe_array.shape[1])]
    ohe_df = pd.DataFrame(ohe_array, columns=columns)
    return ohe_array, ohe_df


def Encode_ms(compositions: [str]):
    return Encode_elements(compositions, possible_elements=sorted(MDefined))


def Encode_nms(compositions: [str]):
    return Encode_elements(compositions, possible_elements=sorted(NMDefined))


def Encode_smiles(smis: [str],
                  nbits=1024,
                  funcname="cluster",
                  AmineTable=None,
                  ):
    if funcname == "dummy":
        ohe_df = pd.get_dummies(smis, dtype=np.float32)
        ohe_array = ohe_df.values.astype(np.float32)
        return ohe_array, ohe_df
    if funcname == "cluster":
        smi_classes = [AmineTable[smi]["saoto_clusterlabel"] for smi in smis]
        ohe_df = pd.get_dummies(smi_classes, dtype=np.float32)
        ohe_array = ohe_df.values.astype(np.float32)
        return ohe_array, ohe_df
    if funcname is None:
        return None, pd.DataFrame()
    if funcname == "mdes":
        mdes_records = [AmineTable[smi] for smi in smis]
        df = pd.DataFrame.from_records(mdes_records)
        numerical_df, obj_df = split_columns(df)
        numerical_columns = numerical_df.columns
        numerical_df = numerical_df.rename(columns={c: "mdes_{}".format(c) for c in numerical_columns})
        return numerical_df.values, numerical_df
    if funcname == "aemdesTA":
        AmineTable_df = pd.read_csv("{}/../DimPredict/3_AEmdes/progression/mdes_bowtie_330.csv".format(this_dir))
        AmineTable_df.set_index("smiles", inplace=True)
        AmineTable = AmineTable_df.to_dict(orient="index")
        mdes_records = [AmineTable[smi] for smi in smis]
        df = pd.DataFrame.from_records(mdes_records)
        numerical_df, obj_df = split_columns(df)
        numerical_columns = numerical_df.columns
        numerical_df = numerical_df.rename(columns={c: "aemdes_{}".format(c) for c in numerical_columns})
        return numerical_df.values, numerical_df
    if funcname == "aemdesTAextra":
        from collections import OrderedDict
        from copy import deepcopy
        from MLModule.utils import VarianceThreshold, load_input_tables
        from sklearn import preprocessing
        import torch
        from MLModule.utils import variance_threshold_selector
        extra_amine_table = load_pkl("{}/../DataGeneration/9_LookupTables/extra_amine_table.pkl".format(this_dir))
        for k in extra_amine_table.keys():
            extra_amine_table[k]["saoto_clusterlabel"] = 0.0  # arbitrary
            extra_amine_table[k]["identifier"] = "extra"  # arbitrary
        atable_with_extra = OrderedDict()
        for d in [AmineTable, extra_amine_table]:
            for k, v in d.items():
                atable_with_extra[k] = v
        amine_df_with_extra = pd.DataFrame.from_dict(atable_with_extra, orient="index")
        amine_df = pd.DataFrame.from_dict(AmineTable, orient="index")

        numerical, obj = split_columns(amine_df)
        mms1 = preprocessing.MinMaxScaler()
        mms1.fit(numerical)
        values = mms1.transform(numerical)
        vt = VarianceThreshold(threshold=1e-5)
        vt.fit(values)
        icols = vt.get_support(indices=True)

        numerical, obj = split_columns(amine_df_with_extra)
        values = mms1.transform(numerical)
        values = values[:, icols]

        # what are the new cols if we use a new vt?
        new_values = preprocessing.MinMaxScaler().fit_transform(numerical)
        new_vt = VarianceThreshold(threshold=1e-5)
        new_vt.fit(new_values)
        new_icols = new_vt.get_support(indices=True)
        used_cols = numerical.columns[icols]
        used_cols_new = numerical.columns[new_icols]
        print([c for c in used_cols_new if c not in used_cols])

        X = deepcopy(values)
        X = torch.from_numpy(X).float()
        net = load_pkl("{}/../DimPredict/3_AEmdes/ae_330.pkl".format(this_dir))
        decoded, encoded = net.forward(X)
        cols = ["encoded_mdes_{}".format(i) for i in range(encoded.shape[1])]
        encoded_df = pd.DataFrame(encoded.cpu().detach().numpy(), columns=cols)
        encoded_df["smiles"] = list(atable_with_extra.keys())
        encoded_df = encoded_df.set_index("smiles")
        encoded_amine_table = encoded_df.to_dict("index")
        mdes_records = [encoded_amine_table[smi] for smi in smis]
        df = pd.DataFrame.from_records(mdes_records)
        numerical_df, obj_df = split_columns(df)
        numerical_columns = numerical_df.columns
        numerical_df = numerical_df.rename(columns={c: "aemdes_{}".format(c) for c in numerical_columns})
        return numerical_df.values, numerical_df

    ohe_array = GetFpArray(smis, nbits, funcname)
    ohe_array = ohe_array.astype(np.float32)
    logging.info("{} gives # of columns: {}".format(inspect.stack()[0][3], ohe_array.shape[1]))
    columns = ["Smiles_bit_{}".format(num) for num in range(ohe_array.shape[1])]
    ohe_df = pd.DataFrame(ohe_array, columns=columns)

    return ohe_array, ohe_df


def encode(dataset, encoding_rules, remove_lowvar=True):
    df = pd.DataFrame()
    for colname in sorted(encoding_rules.keys()):
        encoder, encoder_kws = encoding_rules[colname]
        ohe_array, ohe_df = encoder(dataset.data[colname], **encoder_kws)
        df = pd.concat([df, ohe_df], axis=1)
    if remove_lowvar:
        df = variance_threshold_selector(df, threshold=1e-9)
    return df


def ohe_encode(encoding_rules: dict, condensed_dataset: typing.Union[str, pathlib.Path],
               save_as: typing.Union[str, pathlib.Path]):
    print("---")
    print("encoding:", condensed_dataset)
    Dataset = load_pkl(condensed_dataset)
    df = encode(dataset=Dataset, encoding_rules=encoding_rules)
    Dataset.data = df
    print("encoded shape:", Dataset.data.shape)  # (819, 115)
    save_pkl(Dataset, save_as)
    print("saved as:", save_as)


if __name__ == '__main__':

    sys.stdout = open("./datasets/encoding.log", "w")

    """
    encoding condensed dataset, the result is a pickled dataset with its named defined as:
        dataset_<condensation method>_<amine encoding method>.pkl
    e.g.
        dataset_eab_morgan.pkl
        - condensation is performed on indexing fields: elements, amine, building unit
        - elements field is one-hot encoded
        - building unit field is one-hot encoded
        - amine smiles field is encoded using morgan fp
    """
    AmineTable, BuidTable, ElementTable, IdentifierTable = load_input_tables()

    # none condensed, elements only
    encoding_rules = dict(
        ms=(Encode_ms, dict()),
        nms=(Encode_nms, dict()),
    )
    ohe_encode(
        encoding_rules=encoding_rules,
        condensed_dataset="./condensation/dataset_None.pkl",
        save_as="./datasets/dataset_none_eonly.pkl"
    )

    # none condensed
    encoding_rules = dict(
        ms=(Encode_ms, dict()),
        nms=(Encode_nms, dict()),
        bus=(Encode_bus, dict(BuidTable=BuidTable)),
    )
    ohe_encode(
        encoding_rules=encoding_rules,
        condensed_dataset="./condensation/dataset_None.pkl",
        save_as="./datasets/dataset_none.pkl"
    )

    # eb condensed
    encoding_rules = dict(
        ms=(Encode_ms, dict()),
        nms=(Encode_nms, dict()),
        bus=(Encode_bus, dict(BuidTable=BuidTable)),
    )
    ohe_encode(
        encoding_rules=encoding_rules,
        condensed_dataset="./condensation/dataset_bus-ms-nms.pkl",
        save_as="./datasets/dataset_eb.pkl"
    )

    # eab condensed
    encoding_rules = dict(
        ms=(Encode_ms, dict()),
        nms=(Encode_nms, dict()),
        smiles=(Encode_smiles, dict(nbits=512, funcname="cluster", AmineTable=AmineTable)),
        bus=(Encode_bus, dict(BuidTable=BuidTable)),
    )
    for smiles_encoder_func in (
            'GetHashedAtomPairFingerprintAsBitVect',
            'LayeredFingerprint',
            'GetMACCSKeysFingerprint',
            'GetMorganFingerprintAsBitVect',
            'PatternFingerprint',
            'RDKFingerprint',
            'GetHashedTopologicalTorsionFingerprintAsBitVect',
            "dummy",
            "cluster",
            None,
            "mdes",
            "aemdesTA"
    ):
        encoding_rules["smiles"][1]["funcname"] = smiles_encoder_func
        ohe_encode(
            encoding_rules=encoding_rules,
            condensed_dataset="./condensation/dataset_smiles-bus-ms-nms.pkl",
            save_as="./datasets/dataset_eab_{}.pkl".format(smiles_encoder_func)
        )
