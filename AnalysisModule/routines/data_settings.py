import os
import glob
from collections import namedtuple

"""
a namedtuple for frequently used paths
"""
this_folder = os.path.dirname(os.path.abspath(__file__))

# data is excluded in git
SAOTO_DATA_path = os.path.abspath("{}/../../data".format(this_folder))
# all data folders should have a name like "<category>_data"
# within which the files should look like "<identifier>-<category>.json"

DataSettings = namedtuple("DataSettings", [
    "graph_data",
    "split_data",
    "clean_data",
    "mdes_data",
    "sdes_data",
    "graph_jsons",
    "split_jsons",
    "clean_jsons",
    "mdes_jsons",
    "sdes_jsons",
])  # I have to type them out otherwise fields won't be reg in IDE

DirBaseNames = []
JFileVarNames = []
for field in DataSettings._fields:
    if "_data" in field:
        DirBaseNames.append(field)
    elif "_jsons" in field:
        JFileVarNames.append(field)

data_folder_paths = ["{}/{}".format(SAOTO_DATA_path, dir) for dir in DirBaseNames]
json_file_paths = [list(sorted(
    glob.glob("{}/{}/*-{}.json".format(SAOTO_DATA_path, data_folder_varname, data_folder_varname.split("_")[0])))) for
                   data_folder_varname in DirBaseNames]

SDPATH = DataSettings(*(data_folder_paths + json_file_paths))
