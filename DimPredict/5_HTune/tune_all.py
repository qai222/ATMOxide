from MLModule.fnn_trainers import get_data_and_net
from MLModule.metric import CELossScorer
from MLModule.tuning import Search_Space, saoto_bsearch
from MLModule.utils import SEED
from MLModule.models import Fnn_kargs
from pprint import pprint
import glob
import os

pprint(Search_Space)
for pkl in glob.glob("../4_DatasetEncoding/dataset_*.pkl"):
    DataSet, X_train, X_test, y_train, y_test, net = get_data_and_net(pkl, nn_kargs=Fnn_kargs, split_seed=SEED)
    dataset_name = os.path.basename(pkl)[:-4]
    if os.path.isfile("htune_{}.pkl".format(dataset_name)):
        print("found:", dataset_name)
        continue
    saoto_bsearch(
        SEED, net, CELossScorer, "htune_{}".format(dataset_name), Search_Space, X_train, y_train, use_neptune=True, x_test=X_test, y_test=y_test, tags=[dataset_name, "htune", "2080"]
    )
