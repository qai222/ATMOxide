import json

import pandas as pd

records = pd.read_csv("../3_bulist.csv").to_dict("records")

strange_bus = [
    9,  # PH2O2
    18,  # PO3
    19,  # O-CH3
    21,  # HO-CH2-CH2-OH
    23,  # OH-CH2-CH3
    24,  # PFO3
    25,  # PF2O2
    28,  # Si2O6
    29,  # O-C(O)-CH2-O
    31,  # O2C-CH2-S-CH2-CO2
    32,  # NO3
    34,  # SiO3
    36,  # HCO3
    11,  # AsO3
    6,
    14,
    12,
    3
]

results = dict()
for buid in strange_bus:
    results[buid] = []
    for r in records:
        i = r["identifier"]
        bus = json.loads(r["bus"])
        if buid in bus:
            results[buid].append(i)
    with open("bu-{}.gcd".format(buid), "w") as f:
        for i in results[buid]:
            f.write(i + "\n")
    print("buid:", buid, "len:", len(results[buid]))
