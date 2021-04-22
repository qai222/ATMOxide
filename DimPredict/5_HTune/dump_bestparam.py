from AnalysisModule.routines.util import yaml_filedump
from skopt import load, BayesSearchCV
import glob

for pkl in glob.glob("*.pkl"):
    dsname = pkl.replace(".pkl", "").replace("htune_","")
    opt:BayesSearchCV = load(pkl)
    yaml_filedump(dict(opt.best_params_), "{}.yml".format(dsname))
