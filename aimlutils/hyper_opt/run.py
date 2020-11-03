import warnings
warnings.filterwarnings("ignore")

from aimlutils.utils.gpu import gpu_report
import logging
import optuna
import glob
import yaml
import sys
import os

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

if os.path.isfile(sys.argv[1]):
    with open(sys.argv[1]) as f:
        hyper_config = yaml.load(f, Loader=yaml.FullLoader)
else:
    raise OSError(
        f"Hyperparameter optimization config file {sys.argv[1]} does not exist"
    )
        
if os.path.isfile(sys.argv[2]):
    with open(sys.argv[2]) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
else:
    raise OSError(
        f"Model config file {sys.argv[1]} does not exist"
    )
    
verbose = False
if "log" in hyper_config:
    # Set up a logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    # Stream output to file
    savepath = hyper_config["log"]["save_path"] if "save_path" in hyper_config["log"] else "log.txt"
    mode = "a+" if bool(hyper_config["optuna"]["reload"]) else "w"
    fh = logging.FileHandler(savepath,
                             mode=mode,
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)
    verbose = True
    
# Copy the optuna details to the model config
model_config["optuna"] = hyper_config["optuna"] 
    
# Check if path to objective method exists
if os.path.isfile(model_config["optuna"]["objective"]):
    sys.path.append(model_config["optuna"]["objective"])
    from objective import Objective
else:
    raise OSError(
        f'The objective file {model_config["optuna"]["objective"]} does not exist'
    )

    
# Get the path to save all the data
save_path = model_config["optuna"]["save_path"]
if verbose:
    logging.info(f"Saving optimization details to {save_path}")
    
# Set up the performance metric direction
direction = str(model_config["optuna"]["direction"])
if direction not in ["maximize", "minimize"]:
    raise OSError(
        f"Optimizer direction {direction} not recognized. Choose from maximize or minimize"
    )
if verbose:
    logging.info(f"Direction of optimization {direction}")
    
# Grab the metric
metric = str(model_config["optuna"]["metric"])
if verbose:
    logging.info(f"Using metric {metric}")

# Get list of devices and initialize the Objective class
if bool(model_config["optuna"]["gpu"]):
    gpu_report = sorted(gpu_report().items(), key = lambda x: x[1], reverse = True)
    device = gpu_report[0][0]
else:
    device = 'cpu'
if verbose:
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
storage = storage=f"sqlite:///{cached_study}"

study = optuna.create_study(study_name=study_name,
                            storage=storage,
                            direction=direction,
                            load_if_exists=load_if_exists)
if verbose:
    logging.info(f"Loaded study {study_name} located at {storage}")

# Initialize objective function
objective = Objective(study, model_config, metric, device, verbose)
# Optimize it
study.optimize(objective, n_trials=int(model_config["optuna"]["n_trials"]))
if verbose:
    logging.info(f"Running optimization for {n_trials} trials")

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

if verbose:
    logging.info(f"Saved trial results to {hyper_opt_save_path}")
    logging.info(f"Saved best results to {best_save_path}")