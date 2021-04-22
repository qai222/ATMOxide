import pandas as pd
import ast
from AnalysisModule.routines.util import MDefined, NMDefined, AllElements

def get_accu_abc(mode):
    dfpredall = pd.read_csv("../../DimPredict/7_PbyElement/all_abc.csv")
    pred_records = dfpredall.to_dict("records")
    task_cols = [c for c in dfpredall.columns if c.startswith("task_") and c.endswith(mode)]
    counter = dict()
    for e in AllElements:
        total_prediction = 0
        right_prediction = 0
        for r in pred_records:
            record_elements = ast.literal_eval(r["elements"])
            ms = [ele_symbol for ele_symbol in record_elements if ele_symbol in MDefined]
            nms = [ele_symbol for ele_symbol in record_elements if ele_symbol in NMDefined]
            if e not in nms+ms:
                continue
            for task_col in task_cols:
                total_prediction += 1
                if r[task_col]:
                    right_prediction +=1
        if total_prediction == 0:
            continue
        counter[e] = round(right_prediction/total_prediction * 100, 1)
    zero_elements = ["H", ]
    for e in zero_elements:
        counter.pop(e, None)
    return counter

counter = get_accu_abc("a")
import matplotlib.pyplot as plt
plt.hist(counter.values(), bins=20)
cutoff = 60
nabove_cutoff = 0
for v in counter.values():
    if v > cutoff:
        nabove_cutoff += 1
import numpy as np
print(np.mean(np.array(list(counter.values()))))
print(nabove_cutoff, len(counter.values()), nabove_cutoff/len(counter.values()))
plt.show()
a_counter = counter

counter = get_accu_abc("b")
import matplotlib.pyplot as plt
plt.hist(counter.values(), bins=20)
cutoff = 60
nabove_cutoff = 0
for v in counter.values():
    if v > cutoff:
        nabove_cutoff += 1
import numpy as np
print(np.mean(np.array(list(counter.values()))))
print(nabove_cutoff, len(counter.values()), nabove_cutoff/len(counter.values()))
plt.show()
b_counter = counter

counter = get_accu_abc("c")
import matplotlib.pyplot as plt
plt.hist(counter.values(), bins=20)
cutoff = 90
nabove_cutoff = 0
for v in counter.values():
    if v > cutoff:
        nabove_cutoff += 1
import numpy as np
print(np.mean(np.array(list(counter.values()))))
print(nabove_cutoff, len(counter.values()), nabove_cutoff/len(counter.values()))
plt.show()
c_counter = counter

diff_ab = [a_counter[k]-b_counter[k] for k in a_counter.keys()]
print(diff_ab)
print(min(diff_ab), max(diff_ab))
for k in a_counter.keys():
    if abs(a_counter[k] - b_counter[k]) > 2:
        print(k, a_counter[k], b_counter[k])
