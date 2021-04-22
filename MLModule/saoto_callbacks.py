from skorch import callbacks

from MLModule.metric import DimScorers


# call back functions
def get_callbacks(dimscores=("a", "b", "c"), patience=15, monitor="valid_loss",
                  monitor_lower_is_better=True, ):
    callback_scores = []
    for tov in ("train", "valid"):
        for score_type in dimscores:
            callback_name = "dimscore_{}_{}".format(score_type, tov)
            if tov == "train":
                callback_scores.append(
                    callbacks.EpochScoring(scoring=DimScorers[score_type], lower_is_better=False, name=callback_name,
                                           on_train=True)
                )
            else:
                callback_scores.append(
                    callbacks.EpochScoring(scoring=DimScorers[score_type], lower_is_better=False, name=callback_name,
                                           on_train=False)
                )
    callback_early_stop = callbacks.EarlyStopping(patience=patience, monitor=monitor,
                                                  lower_is_better=monitor_lower_is_better,
                                                  threshold=1e-4,
                                                  threshold_mode='rel',
                                                  )
    callback_functions = callback_scores
    callback_functions.append(callback_early_stop)
    return callback_functions
