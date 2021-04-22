import sys
import pandas as pd
from MLModule.condenser import Condenser


"""
condensation is the process to convert crystal entries to entries with dimv

dimv is a 4d probability vector, (p_0, p_1, p_2, p_3) with sum(dimv) = 1

one has to first define a set of indexing fields (IF) for condensation, e.g.
    - IF = ["identifier"], then condensation is just one-hot encoding the field of `dimensionality` to dimv, no rows will be droped
    - IF = ["elements"], then entries with the same elemental composition will be grouped and converted to one entry with a dimv, in which p_i is calculated as i-count/total-count
IF should be a subset of:
        # ["bus", "ms", "nms"],
        # ["smiles", "bus", "ms", "nms"],
    - identifier
    - elements
    - smiles
    - amine_class
    - M
    - BU
"""

if __name__ == '__main__':
    sys.stdout = open("condensation.log", "w")
    IF_list = [
        ["elements"],
        ["elements", "bus"],
        ["elements", "smiles", "bus"],
        ["bus", "ms", "nms"],
        ["smiles", "bus", "ms", "nms"],
        None,
    ]
    INPUT_DF =pd.read_csv("../../DataGeneration/5_SimpleInput/input.csv")
    for IF in IF_list:
        condenser = Condenser(INPUT_DF, IF)
        try:
            condenser.condense_to_dataset("condensed_{}.pkl".format("-".join(IF)))
        except TypeError:
            condenser.condense_to_dataset("condensed_{}.pkl".format(IF))
