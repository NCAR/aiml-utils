import warnings
warnings.filterwarnings("ignore")

from aimlutils.utils.gpu import gpu_report
import pandas as pd
import logging
import optuna
import time
import glob
import yaml
import sys
import os

def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# References
# https://github.com/optuna/optuna/issues/1365
# https://docs.dask.org/en/latest/setup/hpc.html
# https://dask-cuda.readthedocs.io/en/latest/worker.html
# https://optuna.readthedocs.io/en/stable/tutorial/004_distributed.html#distributed

if len(sys.argv) != 3:
    print(
        "Usage: python main.py hyperparameter.yml model.yml"
    )
    sys.exit()
    

# Set up a logger
root = logging.getLogger()
root.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# Stream output to stdout
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
root.addHandler(ch)

################################################################

# Check if hyperparameter config file exists
if os.path.isfile(sys.argv[1]):
    with open(sys.argv[1]) as f:
        hyper_config = yaml.load(f, Loader=yaml.FullLoader)
else:
    raise OSError(
        f"Hyperparameter optimization config file {sys.argv[1]} does not exist"
    )
    
# Check if the wall-time exists
if "t" not in hyper_config["slurm"]["batch"]:
    raise OSError(
        "You must supply a wall time in the hyperparameter config at slurm:batch:t"
    )
        
# Check if model config file exists
if os.path.isfile(sys.argv[2]):
    with open(sys.argv[2]) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
else:
    raise OSError(
        f"Model config file {sys.argv[1]} does not exist"
    )
    
# Copy the optuna details to the model config
model_config["optuna"] = hyper_config["optuna"] 
    
# Check if path to objective method exists
if os.path.isfile(model_config["optuna"]["objective"]):
    sys.path.append(os.path.split(model_config["optuna"]["objective"])[0])
    from objective import Objective
else:
    raise OSError(
        f'The objective file {model_config["optuna"]["objective"]} does not exist'
    )
    
# Check if the optimization metric direction is supported
direction = str(model_config["optuna"]["direction"])
if direction not in ["maximize", "minimize"]:
    raise OSError(
        f"Optimizer direction {direction} not recognized. Choose from maximize or minimize"
    )
logging.info(f"Direction of optimization {direction}")
    
### Add other checks

################################################################
      
# Stream output to log file
if "log" in hyper_config:
    savepath = hyper_config["log"]["save_path"] if "save_path" in hyper_config["log"] else "log.txt"
    mode = "a+" if bool(hyper_config["optuna"]["reload"]) else "w"
    fh = logging.FileHandler(savepath,
                             mode=mode,
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)
    
# Get the path to save all the data
save_path = model_config["optuna"]["save_path"]
logging.info(f"Saving optimization details to {save_path}")
    
# Grab the metric
metric = str(model_config["optuna"]["metric"])
logging.info(f"Using metric {metric}")

# Get list of devices and initialize the Objective class
if bool(model_config["optuna"]["gpu"]):
    try:
        gpu_report = sorted(gpu_report().items(), key = lambda x: x[1], reverse = True)
        device = gpu_report[0][0]
    except:
        logging.warning(
            "The gpu is not responding to a call from nvidia-smi. Setting gpu device = 0 but this may fail"
        )
        device = 0
else:
    device = 'cpu'
logging.info(f"Using device {device}")

# Initialize the study object
study_name = model_config["optuna"]["name"]
reload_study = bool(model_config["optuna"]["reload"])
cached_study = f"{save_path}/{study_name}"

if not os.path.isfile(cached_study) or not reload_study:
    load_if_exists = False
elif not reload_study:
    os.remove(cached_study)
    load_if_exists = reload_study
else:
    load_if_exists = True

# Initialize the db record and study
storage = f"sqlite:///{cached_study}"

study = optuna.create_study(study_name=study_name,
                            storage=storage,
                            direction=direction,
                            load_if_exists=True)
logging.info(f"Loaded study {study_name} located at {storage}")

# Initialize objective function
objective = Objective(study, model_config, metric, device)

# Optimize it
logging.info(f'Running optimization for {model_config["optuna"]["n_trials"]} trials')
    
wall_time = hyper_config["slurm"]["batch"]["t"]

logging.info(
    f"This script will run for 99% of the wall-time of {wall_time} and try to die without error"
)

wall_time = 0.99 * get_sec(wall_time)
study.optimize(
    objective, 
    n_trials=int(model_config["optuna"]["n_trials"]), 
    timeout = wall_time
)

# Clean up the data files
saved_results = glob.glob(os.path.join(save_path, "hyper_opt_*.csv"))
saved_results = pd.concat(
    [pd.read_csv(x) for x in saved_results], sort = True
).reset_index(drop=True)
saved_results = saved_results.drop(
    columns = [x for x in saved_results.columns if "Unnamed" in x]
)
saved_results = saved_results.sort_values(["trial"]).reset_index(drop = True)
best_parameters = saved_results[saved_results[metric]==max(saved_results[metric])]

# Save results to file
hyper_opt_save_path = os.path.join(save_path, "hyper_opt.csv")
best_save_path = os.path.join(save_path, "best.csv")
saved_results.to_csv(hyper_opt_save_path)
best_parameters.to_csv(best_save_path)

logging.info(f"Saved trial results to {hyper_opt_save_path}")
logging.info(f"Saved best results to {best_save_path}")

# Check a few other stats
pruned_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
]
complete_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]

logging.info(f'Number of requested trials: {model_config["optuna"]["n_trials"]}')
logging.info(f"Number of finished trials: {len(study.trials)}")
logging.info(f"Number of pruned trials: {len(pruned_trials)}")
logging.info(f"Number of complete trials: {len(complete_trials)}")
logging.info(f"Best trial: {study.best_trial.value}")
logging.info("Best parameters in the study:")
for param, val in study.best_params.items():
    logging.info(f"{param}: {val}")
    
if len(study.trials) < model_config["optuna"]["n_trials"]:
    logging.warning(
        "Not all of the trials completed due to the wall-time."
    )
    logging.warning(
        "Set reload = 1 in the hyperparameter config and resubmit some more workers to finish!"
    )

save_study = os.path.join(save_path, f"study_summary.csv")
study.trials_dataframe().to_csv(save_study, index = None)