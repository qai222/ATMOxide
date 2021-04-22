from pprint import pprint
from skopt import load, BayesSearchCV

opt:BayesSearchCV = load("../5_HTune/htune_dataset_eab_aemdes.pkl")
pprint(opt.best_params_)
"""
OrderedDict([('batch_size', 67),
             ('lr', 7.429105354818622e-06),
             ('module__dropout', 0.41095758610670097),
             ('module__h_layers', 1),
             ('module__h_size', 2765)])
"""

opt:BayesSearchCV = load("../5_HTune/htune_dataset_eb_ablation.pkl")
pprint(opt.best_params_)
