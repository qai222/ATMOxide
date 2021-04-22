import copy

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from AnalysisModule.routines.util import save_pkl

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


class Condenser:

    def __init__(self, rawdf: pd.DataFrame, indexing_fields: [str]):
        self.rawdf = rawdf
        self.indexing_fields = indexing_fields

    def condense(self):
        return self.row_merge(self.rawdf, self.indexing_fields)

    @staticmethod
    def is_orthogonal(v):
        return any([x == 1.0 for x in v])

    def condense_to_dataset(self, savefile=None):
        df = self.condense()
        df["orthogonal_dimv"] = [self.is_orthogonal(v) for v in df["dimv"]]
        print("--- CONDENSATION")
        print("condensation with IF:", self.indexing_fields)
        n_orthogonal_vectors = len(df[df["orthogonal_dimv"] == True])
        print("# orthogonal vector:", n_orthogonal_vectors, n_orthogonal_vectors / len(df))
        target = np.zeros((len(df), 4), dtype=np.float32)
        for i in range(len(df)):
            target[i] = df["dimv"].iloc[i]
        if self.indexing_fields is None:
            datadf = df[self.rawdf.columns]
        else:
            datadf = df[self.indexing_fields + ["count_sum"]]
        print("shape in:", self.rawdf.shape)
        print("shape out:", datadf.shape, target.shape)
        print("out columns:", datadf.columns)
        print("out dtypes:", datadf.dtypes)
        data = Bunch(data=datadf, target=target)
        if savefile is not None:
            save_pkl(data, savefile)
        return data

    @staticmethod
    def row_merge(df: pd.DataFrame, indexing_fields: [str]):
        tmp_df = copy.deepcopy(df)
        if indexing_fields is None:
            tmp_df['merge_index'] = list(range(len(tmp_df)))  # we can do this as all list-like fields were sorted
        else:
            tmp_df['merge_index'] = tmp_df[indexing_fields].agg('-'.join,
                                                                axis=1)  # we can do this as all list-like fields were sorted
        new_records = []
        unique = []
        for r in tmp_df.to_dict("records"):
            if r["merge_index"] not in unique:
                degenerate = tmp_df[tmp_df["merge_index"] == r["merge_index"]]
                count_dict = degenerate["dimension"].value_counts(normalize=True).to_dict()
                count_sum = sum(degenerate["dimension"].value_counts(normalize=False).to_dict().values())
                for i in [0, 1, 2, 3]:
                    if i not in count_dict.keys():
                        count_dict[i] = 0.0
                dimvector = [count_dict[k] for k in sorted(count_dict.keys())]
                nr = copy.deepcopy(r)
                nr["dimv"] = dimvector
                nr["count_sum"] = count_sum
                unique.append(r["merge_index"])
                new_records.append(nr)
        return pd.DataFrame.from_records(new_records)
