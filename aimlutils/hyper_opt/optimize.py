import os
import sys
import yaml
import optuna
import logging
import subprocess
from aimlutils.hyper_opt.utils import samplers


def prepare_launch_script(hyper_config, model_config):
    slurm_options = ["#!/bin/bash -l"]
    slurm_options += [
        f"#SBATCH -{arg} {val}" if len(arg) == 1 else f"#SBATCH --{arg}={val}" 
        for arg, val in hyper_config["slurm"]["batch"].items()
    ]
    slurm_options.append("module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1")
    slurm_options.append(f'{hyper_config["slurm"]["kernel"]}')
    import aimlutils.hyper_opt as opt
    aiml_path = os.path.join(
        os.path.abspath(opt.__file__).strip("__init__.py"), 
        "run.py"
    )
    slurm_options.append(f"python {aiml_path} {sys.argv[1]} {sys.argv[2]}")
    return slurm_options

def configuration_report(_dict, path=None):
    if path is None:
        path = []
    for k,v in _dict.items():
        newpath = path + [k]
        if isinstance(v, dict):
            for u in configuration_report(v, newpath):
                yield u
        else:
            yield newpath, v


if __name__ == "__main__":
    
    if len(sys.argv) not in [3, 4]:
        raise "Usage: python main.py hyperparameter.yml model.yml [create database entry only (bool)]"

    if os.path.isfile(sys.argv[1]):
        with open(sys.argv[1]) as f:
            hyper_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise OSError(f"Hyperparameter optimization config file {sys.argv[1]} does not exist")

    if os.path.isfile(sys.argv[2]):
        with open(sys.argv[2]) as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise OSError(f"Model config file {sys.argv[1]} does not exist")
        
    # Override to create the database but skip submitting jobs. This is a debug option so that run.py will run
    create_db_only = True if len(sys.argv) == 4 else False
        
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
    if "log" in hyper_config:
        savepath = hyper_config["log"]["save_path"] if "save_path" in hyper_config["log"] else "log.txt"
        mode = "a+" if bool(hyper_config["optuna"]["reload"]) else "w"
        fh = logging.FileHandler(savepath,
                                 mode=mode,
                                 encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)
        
    # Print the configurations to the logger
    logging.info("Current hyperparameter configuration settings:")
    for p, v in configuration_report(hyper_config):
        full_path = ".".join(p)
        logging.info(f"{full_path}: {v}")
    logging.info("Current model configuration settings:")
    for p, v in configuration_report(model_config):
        full_path = ".".join(p)
        logging.info(f"{full_path}: {v}")

    # Set up new db entry if reload = 0 
    reload_study = bool(hyper_config["optuna"]["reload"])
        
        
    # Check if save directory exists
    if not os.path.isdir(hyper_config["optuna"]["save_path"]):
        raise OSError(
            f'Create the save directory {hyper_config["optuna"]["save_path"]} and try again'
        )
        
    # Initiate a study for the first time
    if not reload_study:        
        name = hyper_config["optuna"]["name"]
        path_to_study = os.path.join(hyper_config["optuna"]["save_path"], name)
        storage =f"sqlite:///{path_to_study}"
        
        if os.path.isfile(path_to_study):
            message = f"The study already exists at {path_to_study} and reload was False."
            message += f" Delete the study at {path_to_study} and try again"
            raise OSError(message)
        
        direction = hyper_config["optuna"]["direction"]
        if direction not in ["maximize", "minimize"]:
            raise OSError(
                f"Optimizer direction {direction} not recognized. Choose from maximize or minimize"
            )
        
        if "sampler" not in hyper_config["optuna"]:
            sampler = optuna.samplers.TPESampler()
        else:
            sampler = samplers(hyper_config["optuna"]["sampler"])
        
        create_study = optuna.create_study(
            study_name = name,
            storage = storage,
            direction = direction,
            sampler = sampler
        )
        
    # Stop here if 4th arg is defined -- intention is that you manually run run.py for debugging purposes
    if create_db_only:
        sys.exit()
        
    # Prepare slurm script
    launch_script = prepare_launch_script(hyper_config, model_config)
    
    # Save the configured script
    script_path = os.path.split(hyper_config["optuna"]["save_path"])[0]
    script_location = os.path.join(script_path, "launch.sh")
    with open(script_location, "w") as fid:
        for line in launch_script:
            fid.write(f"{line}\n")
    raise

    # Launch the jobs
    job_ids = []
    name_condition = "J" in hyper_config["slurm"]["batch"]
    slurm_job_name = hyper_config["slurm"]["batch"]["J"] if name_condition else "hyper_opt"
    n_workers = hyper_config["slurm"]["jobs"]
    for worker in range(n_workers):
        w = subprocess.Popen(
            f"sbatch -J {slurm_job_name}_{worker} {script_location}", 
            shell=True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE
        ).communicate()
        job_ids.append(
            w[0].decode("utf-8").strip("\n").split(" ")[-1]
        )
        logging.info(
            f"Submitted batch job {worker + 1}/{n_workers} with id {job_ids[-1]}"
        )
        
    # Write the job ids to file for reference
    with open(os.path.join(script_path, "job_ids.txt"), "w") as fid:
        for line in job_ids:
            fid.write(f"{line}\n")