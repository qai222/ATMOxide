import json

import pandas as pd

from AnalysisModule.prepare.diagram import BuildingUnit
from AnalysisModule.routines.util import read_jsonfile

"""
4 ways to deal with strange bus
A - exclude bu, keep crystals
A' - merge bu, keep crystals
B - exclude crystals
C - keep all

note 2020/11/24:
    - all A are modified to A'
    - use bu_0 bu_1 bu_2 to represent bus
    - if bu_x does not exist, the field is set to -1


buid: 9 len: 9 C
    - H2PO2, reactants include hypophosphorous acid
buid: 18 len: 9 A'
    - this is tricky: some of them come from PO4 across pbc, some of them are HPO3 without hydrogens
    - HPO3: (BU0)
        BEZVIO
        BEZVOU
        BEZVUA
        BEZWAH
        CASWIE
        TEXSEV
    - PO4: (BU1)
        CUHCIR
        POVMOC
        QOBWEJ
buid: 19 len: 21 B
    - some of them are MeOH e.g. coordinated to a metal, some of them are MeO with O acting as a bridge e.g. between metals
buid: 21 len: 5 B
    - ethylene glycol
buid: 23 len: 8 B
    - ethanol
buid: 24 len: 9 A (2020/11/24 -> 1 A', BU1)
    - PFO3
    - similar to BU25, HPF6 is used as input
buid: 25 len: 2 A (2020/11/24 -> 1 A', BU1)
    - PF2O2
    - HPF6 involved in both synthesis
buid: 28 len: 1 A (2020/11/24 -> 1 A', BU5)
    - octahedral SiO6(2-)
buid: 29 len: 1 B
    - O-C(O)-CH2-O
    - synthesis uses glycolic acid 
buid: 31 len: 7 B
    - O2C-CH2-S-CH2-CO2
    - uranyl thiodigycolate is used in synthesis
buid: 32 len: 1 A'
    - KAPSUR, distorted NO3 (BU10)
buid: 34 len: 1 A'
    - SiO3, just broken SiO4 (BU5) by pbc
buid: 36 len: 1 A'
    - WAQVOZ, glitched CO3 (BU15)
"""

records = pd.read_csv("3_bulist.csv").to_dict("records")
curated_records = []
curated_bus = []
for ir in range(len(records)):
    records[ir]["bus"] = json.loads(records[ir]["bus"])
    identifier = records[ir]["identifier"]
    # merge A'
    if 18 in records[ir]["bus"]:
        if identifier in ["BEZVIO", "BEZVOU", "BEZVUA", "BEZWAH", "CASWIE", "TEXSEV", ]:
            records[ir]["bus"] = [0 if x == 18 else x for x in records[ir]["bus"]]
        elif identifier in ["CUHCIR", "POVMOC", "QOBWEJ", ]:
            records[ir]["bus"] = [1 if x == 18 else x for x in records[ir]["bus"]]
        else:
            raise NotImplementedError("BU 18 not merged due to unknown identifier: {}".format(identifier))
    # aprime_dict = {32: 10, 34: 5, 36: 15}
    aprime_dict = {32: 10, 34: 5, 36: 15, 24: 1, 25: 1, 28: 5}  # 2020/11/24
    for buid_aprime in aprime_dict.keys():
        if buid_aprime in records[ir]["bus"]:
            records[ir]["bus"] = [aprime_dict[buid_aprime] if x == buid_aprime else x for x in records[ir]["bus"]]
    # exclude crystals with B
    if set(records[ir]["bus"]).intersection({19, 21, 23, 29, 31}):
        continue

    # very few crystal has more than 2 bus
    if len(records[ir]["bus"]) > 2:
        print("bus len > 2: ", len(records[ir]["bus"]), records[ir]["identifier"])
    curated_bus += records[ir]["bus"]
    curated_records.append(records[ir])

df = pd.DataFrame.from_records(curated_records)
df.to_csv("4_bucurate.csv", index=False)
df[["identifier"]].to_csv("4_bucurate.gcd", index=False, header=False)

curated_bus = set(curated_bus)
print("# of curated bus", len(curated_bus))
bus = read_jsonfile("3_bulist.json")
outf = open("4_bucurate.html", "w")
for bu in bus:
    bu: BuildingUnit
    if bu.buid in curated_bus:
        svgtext = bu.draw_by_cdg(title=bu.buid)
        outf.write(svgtext)
outf.close()
