import os

import neptune
import neptunecontrib.monitoring.skopt as sk_utils
from skopt import BayesSearchCV
from skopt import dump
from skopt.space import Real, Integer

from MLModule.metric import DimScoreModes, DimScore
from MLModule.utils import set_seed, neptune_api_token, neptune_proj_name

Search_Space = {
    'module__dropout': Real(0.0, 0.5, prior="uniform"),
    'module__h_size': Integer(20, 4000, prior="uniform"),
    'module__h_layers': Integer(1, 4, prior="uniform"),
    'batch_size': Integer(32, 256, prior="uniform"),
    'lr': Real(5e-6, 1e-4, prior='log-uniform'),
}


def saoto_bsearch(seed, neural_net, scoring, expt_name: str, search_space: dict, x_train, y_train, wdir="./",
                  use_neptune=True, x_test=None, y_test=None, tags: [str] = []):
    set_seed(seed)
    neural_net.verbose = 0
    bsearch_params = dict(
        n_iter=40,
        cv=5,
        scoring=scoring,
        n_points=1,
        n_jobs=5,
        verbose=10,
        random_state=seed,
        return_train_score=True,
        pre_dispatch=5,
    )
    # gpu mem < 8GB
    if use_neptune:
        neptune.init(api_token=neptune_api_token, project_qualified_name=neptune_proj_name)
        experiment = neptune.create_experiment(
            name=expt_name,
            params=bsearch_params, tags=tags)
        neptune_callback = sk_utils.NeptuneCallback(experiment=experiment)

    whereami = os.path.abspath(os.getcwd())
    os.chdir(wdir)

    opt = BayesSearchCV(
        neural_net,
        search_spaces=search_space,
        **bsearch_params,
    )

    if use_neptune:
        opt.fit(x_train, y_train, callback=[neptune_callback])
    else:
        opt.fit(x_train, y_train)

    if x_test is not None and y_test is not None:
        y_pred = opt.predict(x_test)
        for mode in DimScoreModes.keys():
            dims = DimScore(y_test, y_pred, mode)
            print("test_dimscore_{}".format(mode), dims)

    dump(opt, "{}.pkl".format(expt_name))
    if use_neptune:
        sk_utils.log_results(opt.optimizer_results_[-1], experiment=experiment)
        neptune.stop()
    os.chdir(whereami)
