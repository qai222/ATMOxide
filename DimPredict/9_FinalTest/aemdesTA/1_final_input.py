import re
from rdkit.Chem import CanonSmiles

import pandas as pd
from pymatgen.core.composition import Composition

from AnalysisModule.prepare.diagram import BuildingUnit
from AnalysisModule.routines.util import MDefined, read_jsonfile, get_no_hydrogen_composition, find_nms, find_ms
from MLModule.encoding import load_input_tables

bulist: [BuildingUnit] = read_jsonfile("../../../DataGeneration/1_ChemicalDiagramSearch/3_bulist.json")

amine_translate_df = pd.read_csv("../amine_translate.csv")
abbrs = amine_translate_df["abbr"].tolist()
smis = amine_translate_df["smiles"].tolist()
amine_translate_table = dict(zip(abbrs, smis))


AmineTable, BuidTable, ElementTable, IdentifierTable = load_input_tables()
# bu:BuildingUnit
# for bu in bulist:
#     if "Se" in bu.composition.formula:
#         print(bu.composition)
def find_bu_by_comp(c: Composition):
    for b in bulist:
        if b.composition == c:
            return b
    for b in bulist:
        if b.composition == c.reduced_composition:
            return b


def get_bus_from_comp_formula(cf: str):
    """
    if it has "()", extract things within and find it in bulist, if cannot find exact match, strip hydrogen and find it
    if it does not have "()", search it in bulist directly
    """
    bus_found = []
    if "(" in cf or ")" in cf:
        possible_bus = re.findall(r'\((.*?)\)', cf)
        for pbu in possible_bus:
            if "(" in pbu or ")" in pbu:
                # print(pbu)  # always good to check
                continue
            pbuc = Composition(pbu)
            bu_found = find_bu_by_comp(pbuc)
            if bu_found is None:
                pbuc = get_no_hydrogen_composition(pbuc)
                bu_found = find_bu_by_comp(pbuc)
            if bu_found is None:
                continue
            bus_found.append(bu_found)
    else:
        pbuc = Composition(cf)
        bu_found = find_bu_by_comp(pbuc)
        if bu_found is None:
            pbuc = get_no_hydrogen_composition(pbuc)
            bu_found = find_bu_by_comp(pbuc)
        if bu_found is None:
            return []
        bus_found.append(bu_found)
    return sorted(set([b.buid for b in bus_found]))


def process_rawformula(f: str, replace: dict = {"ox": "C2O4", }):
    f = f.split("·")[0]
    f = f.split(".")[0]
    for k, v in replace.items():
        f = f.replace(k, v)
    comps = re.findall(r'\[(.*?)\]', f)
    mocomp = Composition()
    bus = []
    for c in comps:
        bus += get_bus_from_comp_formula(c)
        thiscomp = Composition(c)
        thiscomp_elements = [e.symbol for e in thiscomp]
        if set(thiscomp_elements).intersection(MDefined):
            mocomp += thiscomp
    return [e.symbol for e in mocomp.elements], bus


df = pd.read_csv("../formatted.csv", header=None)
records = df.to_dict("records")
formatted_records = []
for r in records:
    isinclude = r[8].lower()
    if isinclude.startswith("ex"):
        continue
    dimraw = r[3]
    formula = r[1]
    codename = r[0]
    dim = int(dimraw[0])
    elements, bus = process_rawformula(formula)
    if "H" in elements:
        elements.remove("H")
    ms = find_ms(elements)
    nms = find_nms(elements)
    abbr = r[4]
    if abbr == "3-apry":
        abbr = "3-apyr"
    if abbr == "2-mpip/en":  # 2 types of amine
        continue
    smiles = amine_translate_table[abbr]
    try:
        smiles = CanonSmiles(smiles, 0)
    except:
        print("canon smiles failed:", abbr, codename)
        smiles = amine_translate_table[abbr]
    if smiles not in AmineTable.keys():
        print("not in csd:", abbr, codename, smiles)
    fr = dict(dimension=dim, elements=elements, bus=bus, nms=nms, ms=ms, smiles=smiles)
    formatted_records.append(fr)
fdf = pd.DataFrame.from_records(formatted_records)
fdf.to_csv("1_final_input.csv", index=False)
