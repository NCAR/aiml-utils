import os
import sys
import yaml
import optuna
import logging
import pandas as pd

if len(sys.argv) != 2:
    raise OSError(
        "Usage: python report.py hyperparameter.yml"
    )

# Check if hyperparameter config file exists
if os.path.isfile(sys.argv[1]):
    with open(sys.argv[1]) as f:
        hyper_config = yaml.load(f, Loader=yaml.FullLoader)
else:
    raise OSError(
        f"Hyperparameter optimization config file {sys.argv[1]} does not exist"
    )
    
# Set up a logger
root = logging.getLogger()
root.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# Stream output to stdout
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
root.addHandler(ch)

save_path = hyper_config["optuna"]["save_path"]
study_name = hyper_config["optuna"]["name"]
reload_study = bool(hyper_config["optuna"]["reload"])
cached_study = f"{save_path}/{study_name}"
    
storage = f"sqlite:///{cached_study}"

study = optuna.load_study(study_name=study_name, storage=storage)

# Check a few other stats
pruned_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
]
complete_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]

logging.info(f'Number of requested trials: {hyper_config["optuna"]["n_trials"]}')
logging.info(f"Number of finished trials: {len(study.trials)}")
logging.info(f"Number of pruned trials: {len(pruned_trials)}")
logging.info(f"Number of complete trials: {len(complete_trials)}")
logging.info(f"Best trial: {study.best_trial.value}")
logging.info("Best parameters in the study:")
for param, val in study.best_params.items():
    logging.info(f"{param}: {val}")
    
if len(study.trials) < hyper_config["optuna"]["n_trials"]:
    logging.warning(
        "Not all of the trials completed due to the wall-time."
    )
    logging.warning(
        "Set reload = 1 in the hyperparameter config and resubmit some more workers to finish!"
    )
    
study.trials_dataframe().to_csv(f"{study_name}.csv", index = None)