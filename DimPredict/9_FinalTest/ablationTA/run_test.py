from MLModule.fnn_trainers import reload_net
from AnalysisModule.routines.util import load_pkl
from MLModule.metric import DimScoreModes, DimScore

dataset_final = load_pkl("./dataset_eab_none.pkl")
dataset = load_pkl("../../4_DatasetEncoding/dataset_eab_None.pkl")
nsamples = len(dataset_final.data) - len(dataset.data)

dataset_final.data = dataset_final.data.tail(nsamples)
dataset_final.target = dataset_final.target[-nsamples:]
net = reload_net("../../6_TrainOpt/dataset_eab_None_init_params.pkl", "../../6_TrainOpt/dataset_eab_None_params.pkl")
y_pred = net.predict(dataset_final.data.values)
y_test = dataset_final.target

import sys
sys.stdout = open("final_test.log", "w")
for m in DimScoreModes:
    dims = DimScore(y_test, y_pred, m)
    print(m, dims, dims*len(y_test), len(y_test))
