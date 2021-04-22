import os
import re

from AnalysisModule.prepare.saentry import SaotoEntry
from AnalysisModule.routines.data_settings import SDPATH
from AnalysisModule.routines.util import Structure, MDefined
from AnalysisModule.routines.util import get_mo_compositions_from_csd, read_jsonfile, get_mo_compositions_from_sslist

thisdir = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
from ccdc.io import EntryReader, Entry

csd_reader = EntryReader('CSD')

identifiers = pd.read_csv("{}/../1_ChemicalDiagramSearch/4_bucurate.csv".format(thisdir))["identifier"].tolist()


def compcheck_pass(logfile):
    compcheckpass = None
    with open(logfile, "r") as f:
        fs = f.read()
        if "compcheck failed" in fs:
            compcheckpass = False
        elif "compcheck good" in fs:
            compcheckpass = True
    return compcheckpass


def get_sae_structures(identifier):
    clean_json = "{}/{}-clean.json".format(SDPATH.clean_data, identifier)
    sae: SaotoEntry = read_jsonfile(clean_json)
    return Structure.from_dict(sae.disodered_structure), Structure.from_dict(sae.clean_structure)


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
    return " ".join(ms + anions + sorted(symbols))


def identifier2elements_sslit(identifier, exclude_hydrogens=True):
    split_json_file = "{}/{}-split.json".format(SDPATH.split_data, identifier)

    if not os.path.isfile(split_json_file) or os.path.getsize(split_json_file) == 0:
        raise FileNotFoundError('split json not found: {}'.format(identifier))

    results = read_jsonfile(split_json_file)
    disorder_ss_list = results['disorder_structure_substructure_list']
    c = get_mo_compositions_from_sslist(disorder_ss_list)
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
    return " ".join(ms + anions + sorted(symbols))


def get_raw_cif(identifier):
    clean_json = "{}/{}-clean.json".format(SDPATH.clean_data, identifier)
    sae: SaotoEntry = read_jsonfile(clean_json)
    return sae.details["cifstring"]


def identifier2record(identifier):
    sdes_json = "{}/{}-sdes.json".format(SDPATH.sdes_data, identifier)
    sdes_log = "{}/{}-sdes.log".format(SDPATH.sdes_data, identifier)

    record = dict(identifier=identifier, comment="")

    if not os.path.isfile(sdes_json):
        record["comment"] += "sdes failed, check manually"
        return record

    for k, v in read_jsonfile(sdes_json).items():
        record["SDES_" + k] = v

    compcheck = compcheck_pass(sdes_log)
    if compcheck is None:
        record["comment"] += "compcheck not triggered, check manually"
    elif compcheck is False:
        if identifier2elements_sslit(identifier) == identifier2elements_csd(identifier):
            record["comment"] += "compcheck false, but compelementcheck ok, verify needed"
        else:
            if record["SDES_Dimension"] == 0:
                record["comment"] += "compcheck false, compelementcheck faild, but dimension is zero, verify needed"
            else:
                record["comment"] += "compcheck false, compelementcheck faild, but dimension is nonzero, check manually"
    return record


rs = []
verify_rs = []
check_rs = []
for identifier in identifiers:
    r = identifier2record(identifier)
    comment = r["comment"]
    if "check manually" in comment:
        check_rs.append(r)
    elif "verify needed" in comment:
        verify_rs.append(r)
    else:
        rs.append(r)

print("good", len(rs))
print("to be verified", len(verify_rs))
print("to be checked", len(check_rs))
df = pd.DataFrame.from_records(rs + verify_rs + check_rs)
print("total", len(df))
df.to_csv("2_sdes.csv", index=False)
# good 3187
# to be verified 304
# to be checked 236
# total 3727


for r in check_rs:
    if "SDES_Dimension" not in r.keys():
        r["SDES_Dimension"] = "*"
    else:
        r["SDES_Dimension"] = str(r["SDES_Dimension"] )


check_rs = sorted(check_rs, key=lambda x:x["SDES_Dimension"])
df_check = pd.DataFrame.from_records(check_rs)
df_check["check"] = ["*"] * len(df_check)
df_check = df_check[["identifier", "SDES_Dimension", "check"]]
df_check.to_csv("2_check.origin", index=False, sep=" ")
# generates a csv for manually inspecting dimension
# most of them are due to alkaline metals: e.g. counter ion Na+ is bonded to a set of cages since Na bond cutoff is
# based on atomic radius not cation radius
