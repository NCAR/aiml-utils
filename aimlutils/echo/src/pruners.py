import warnings
warnings.filterwarnings("ignore")

import sys
import optuna
import logging
from tensorflow.keras.callbacks import Callback


logger = logging.getLogger(__name__)


class KerasPruningCallback(Callback):

    def __init__(self, trial, monitor, interval = 1):
        # type: (optuna.trial.Trial, str) -> None

        super(KerasPruningCallback, self).__init__()

        self.trial = trial
        self.monitor = monitor
        self.interval = interval

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
