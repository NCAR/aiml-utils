import warnings
warnings.filterwarnings("ignore")

import sys
import optuna
import logging
from tensorflow.keras.callbacks import Callback


logger = logging.getLogger(__name__)


supported_trials = [
    "categorical",
    "discrete_uniform",
    "float",
    "int",
    "loguniform",
    "uniform"
]


def trial_suggest_loader(trial, config):
    
    try:
    
        _type = config["type"]
        if _type == "categorical":
            return int(trial.suggest_categorical(**config["settings"]))
        elif _type == "discrete_uniform":
            return int(trial.suggest_discrete_uniform(**config["settings"]))
        elif _type == "float":
            return float(trial.suggest_float(**config["settings"]))
        elif _type == "int":
            return int(trial.suggest_int(**config["settings"]))
        elif _type == "loguniform":
            return float(trial.suggest_loguniform(**config["settings"]))
        elif _type == "uniform":
            return float(trial.suggest_uniform(**config["settings"]))
        else: #if _type not in supported_trials:
            message = f"Type {_type} is not valid. Select from {supported_trials}"
            logger.warning(message)
            raise OSError(message)
            
    except Exception as E:
        print("FAILED IN TRIAL SUGGEST", E, config)
        raise OSError
    

supported_samplers = [
    "TPESampler",
    "GridSampler",
    "RandomSampler",
    "CmaEsSampler",
    "IntersectionSearchSpace"
]

def samplers(sampler):
    _type = sampler["type"]
    if _type not in supported_samplers:
        message = f"Sampler {_type} is not valid. Select from {supported_samplers}"
        logger.warning(message)
        raise OSError(message)
    if _type == "TPESampler":
        return optuna.samplers.TPESampler()
    elif _type == "GridSampler":
        if "search_space" not in sampler:
            raise OSError("You must provide search_space options with the GridSampler.")
        else:
            return optuna.samplers.GridSampler(sampler["search_space"])
    elif _type == "RandomSampler":
        return optuna.samplers.RandomSampler()
    elif _type == "CmaEsSampler":
        return optuna.integration.CmaEsSampler()
    elif _type == "IntersectionSearchSpace":
        return optuna.integration.IntersectionSearchSpace()
    

class KerasPruningCallback(Callback):

    def __init__(self, trial, monitor):
        # type: (optuna.trial.Trial, str) -> None

        super(KerasPruningCallback, self).__init__()

        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        # type: (int, Dict[str, float]) -> None

        logs = logs or {}
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.structs.TrialPruned(message)