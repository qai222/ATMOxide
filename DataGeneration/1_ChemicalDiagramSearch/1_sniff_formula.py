import re

import pandas as pd
from ccdc.io import EntryReader, Entry
from tqdm import tqdm

from AnalysisModule.routines.util import MDefined, csdformula2pmg_composition, Composition

CSD_READER = EntryReader('CSD')


def composition_symbols(c: Composition):
    return set(e.symbol for e in c.elements)


def get_identifier_header(i: str):
    return re.findall(r"[A-z]+", i)[0]


def sniff_filter(entry: Entry):
    return entry.has_3d_structure


def formula_filter(entry: Entry):
    formula = entry.formula
    elements = set(re.findall(r"[A-Z]{1}[a-z]{0,1}", formula))
    if not (elements.intersection(MDefined) and {"C", "N", "O"}.issubset(elements)):
        return False
    compositions = csdformula2pmg_composition(formula)
    if len(compositions) < 2:
        return False

    composition_symbol_list = list(composition_symbols(c) for c in compositions)
    if not any(cs == {"C", "N", "H"} for cs in composition_symbol_list):
        return False
    if not any(cs.intersection(MDefined) for cs in composition_symbol_list):
        return False
    return True


if __name__ == '__main__':
    """
    remove 1. no 3d structures 2. duplicating identifiers e.g. "ABC" and "ABC01"
    """
    iheader_set = set()
    output_identifiers = []
    for entry in tqdm(CSD_READER):
        if sniff_filter(entry) and formula_filter(entry):
            iheader = get_identifier_header(entry.identifier)
            if iheader not in iheader_set:
                iheader_set.add(iheader)
                output_identifiers.append(entry.identifier)

    df = pd.DataFrame(sorted(output_identifiers), columns=['identifier'])
    df.to_csv("1_sniff_formula.gcd", index=False, header=False)
