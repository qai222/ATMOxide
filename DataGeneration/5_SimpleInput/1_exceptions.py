import pandas as pd
from AnalysisModule.routines.data_settings import SDPATH
import os

thisdir = os.path.dirname(os.path.abspath(__file__))

identifiers = pd.read_csv("{}/../1_ChemicalDiagramSearch/4_bucurate.csv".format(thisdir))["identifier"].tolist()
for i in identifiers:
    clean_json = "{}/{}-clean.json".format(SDPATH.clean_data, i)
    split_json = "{}/{}-split.json".format(SDPATH.split_data, i)
    sdes_json = "{}/{}-sdes.json".format(SDPATH.sdes_data, i)
    if not os.path.isfile(clean_json):
        print("clean failed: {}".format(i))
    if not os.path.isfile(split_json):
        print("split failed: {}".format(i))
    if not os.path.isfile(sdes_json):
        print("sdes failed: {}".format(i))

