import os

import pandas as pd
from ccdc.io import EntryReader, Entry
from tqdm import tqdm

from AnalysisModule.routines.data_settings import SDPATH
from AnalysisModule.routines.util import get_mo_compositions_from_csd, read_jsonfile, get_mo_compositions_from_sslist, MDefined

"""
in phase one we have the following:
    - identifiers, smiles, bus from ChemicalDiagramSearch
    - dimension from StructDes
"""

csd_reader = EntryReader('CSD')
tqdm.pandas()

import re


def is_deuterium(symbol):
    try:
        return re.findall("[a-zA-Z]+", symbol)[0].upper() == "D"
    except IndexError:
        return False


def identifier2elements_csd(identifier, exclude_hydrogens=True):
    entry: Entry = csd_reader.entry(identifier)
    c = get_mo_compositions_from_csd(entry.formula)
    ms = []
    anions = []
    symbols = []
    for e in c.elements:
        if is_deuterium(e.symbol):
            if exclude_hydrogens:
                continue
            else:
                symbols.append("H")
        if e.symbol in MDefined:
            ms.append(e.symbol)
        elif e.symbol == "O":
            anions.append(e.symbol)
        else:
            if e.symbol == "H" and exclude_hydrogens:
                continue
            symbols.append(e.symbol)
    return sorted(ms + anions + symbols)

def identifier2elements_csd_with_stoichiometry(identifier):
    entry: Entry = csd_reader.entry(identifier)
    c = get_mo_compositions_from_csd(entry.formula)
    return c.formula

def identifier2csdformula(identifier):
    """just the formula field"""
    entry: Entry = csd_reader.entry(identifier)
    return entry.formula


def identifier2elements_sslit(identifier, exclude_hydrogens=True):
    split_json_file = "{}/{}-split.json".format(SDPATH.split_data, identifier)

    if not os.path.isfile(split_json_file) or os.path.getsize(split_json_file) == 0:
        raise FileNotFoundError('split json not found: {}'.format(identifier))

    results = read_jsonfile(split_json_file)
    disorder_ss_list = results['disorder_structure_substructure_list']
    c = get_mo_compositions_from_sslist(disorder_ss_list)
    metals = []
    anions = []
    symbols = []
    for e in c.elements:
        if is_deuterium(e.symbol):
            if exclude_hydrogens:
                continue
            else:
                symbols.append("H")
        if e.is_metal or e.is_metalloid:
            metals.append(e.symbol)
        elif e.symbol == "O":
            anions.append(e.symbol)
        else:
            if e.symbol == "H" and exclude_hydrogens:
                continue
            symbols.append(e.symbol)
    return sorted(metals + anions + symbols)


def find_ms(symbols: [str]):
    ms = []
    for s in symbols:
        if s in MDefined:
            ms.append(s)
    return sorted(ms)


def find_nms(symbols: [str]):
    ms = []
    for s in symbols:
        if s not in MDefined:
            ms.append(s)
    return sorted(ms)


def m_mask(elements: str):
    """Mg Ca A Fe B C --> M A B C"""
    symbols = elements.split()
    hasm = False
    nonmetal = []
    for s in symbols:
        if s in MDefined:
            hasm = True
        else:
            nonmetal.append(s)
    if hasm:
        masked = "M " + " ".join(sorted(nonmetal))
    else:
        masked = " ".join(sorted(nonmetal))
    return masked


if __name__ == '__main__':
    thisdir = os.path.dirname(os.path.abspath(__file__))

    dim_df = pd.read_csv("{}/../3_SDes/3_dim.csv".format(thisdir))
    print("read dimension df:", dim_df.shape)
    bu_df = pd.read_csv("{}/../1_ChemicalDiagramSearch/4_bucurate.csv".format(thisdir))
    print("read identifier, smiles, bus df:", bu_df.shape)

    df_in = dim_df.merge(bu_df, on="identifier")
    print("merged input df:", df_in.shape)

    # # note: identifier2elements_sslit() may give results different from identifier2elements_csd()
    # # use the csd one to be safe
    # df_in['elements'] = df_in.progress_apply(lambda x: identifier2elements_sslit(x["identifier"]), axis=1)
    df_in['elements'] = df_in.progress_apply(lambda x: identifier2elements_csd(x["identifier"]), axis=1)
    # df_in['masked_elements'] = df_in.apply(lambda x: m_mask(x["elements"]), axis=1)

    df_in['ms'] = df_in.apply(lambda x: find_ms(x["elements"]), axis=1)
    df_in['nms'] = df_in.apply(lambda x: find_nms(x["elements"]), axis=1)
    df_in.to_csv("input.csv", index=False)


    # # if you need stoi
    # df_in['mo_formula'] = df_in.progress_apply(lambda x: identifier2elements_csd_with_stoichiometry(x["identifier"]),
    #                                          axis=1)
    # df_in['csdformula'] = df_in.progress_apply(lambda x: identifier2csdformula(x["identifier"]),
    #                                               axis=1)
    # df_in.to_csv("input_stoi.csv", index=False)
