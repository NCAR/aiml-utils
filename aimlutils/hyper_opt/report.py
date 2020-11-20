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

#storage = f'postgresql+psycopg2://john:schreck@localhost/{cached_study}'
storage = f"sqlite:///{cached_study}"

study = optuna.load_study(study_name=study_name, storage=storage)
#study._storage = study._storage._backend  # avoid using chaed storage

# Check a few other stats
pruned_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
]
complete_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]

logging.info(f'Number of requested trials per worker: {hyper_config["optuna"]["n_trials"]}')
logging.info(f"Number of trials in the database: {len(study.trials)}")
logging.info(f"Number of pruned trials: {len(pruned_trials)}")
logging.info(f"Number of completed trials: {len(complete_trials)}")

if len(complete_trials) == 0:
    logging.info("There are no complete trials in this study.")
    logging.info("Wait until the workers finish a few trials and try again.")
    sys.exit()

logging.info(f"Best trial: {study.best_trial.value}")

importance = optuna.importance.get_param_importances(study=study)
logging.info(f"Parameter importance {dict(importance)}")

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
    
save_fn = os.path.join(save_path, f"{study_name}.csv")
logging.info(f"Saving the results of the study to file at {save_fn}")
study.trials_dataframe().to_csv(save_fn, index = None)