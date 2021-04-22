from AnalysisModule.routines.util import load_pkl
from MLModule.reapply import reapply_net
from MLModule.utils import SEED
import sys

sys.stdout = open("transfer.log", "w")

if __name__ == '__main__':
    old_initp = load_pkl("../6_TrainOpt/dataset_eab_None_init_params.pkl")
    reapply_net(old_initp, "../6_TrainOpt/dataset_eab_None_params.pkl", "../4_DatasetEncoding/dataset_eb_ablation.pkl",
                splitseed=SEED, oldnetname="eab_None", newnetname="eb_ablation")
