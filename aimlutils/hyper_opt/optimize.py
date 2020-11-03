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
    slurm_options.append("ncar_pylib")
    slurm_options.append(f"python run.py {sys.argv[1]} {sys.argv[2]}")
    return slurm_options


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        raise "Usage: python main.py hyperparameter.yml model.yml"

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

    # Set up new db entry if reload = 0 
    reload_study = bool(hyper_config["optuna"]["reload"])
        
    if not reload_study:        
        name = hyper_config["optuna"]["name"]
        path_to_study = os.path.join(hyper_config["optuna"]["save_path"], name)
        storage =f"sqlite:///{path_to_study}"
        
        if os.path.isfile(path_to_study):
            message = f"WARNING: The study already exists at {path_to_study}."
            message += f" You must delete {path_to_study} before proceeding."
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
        
    # Prepare slurm script
    launch_script = prepare_launch_script(hyper_config, model_config)
    
    # Save the configured script
    script_path = os.path.split(hyper_config["log"]["save_path"])[0]
    script_location = os.path.join(script_path, "launch.sh")
    with open(script_location, "w") as fid:
        for line in launch_script:
            fid.write(f"{line}\n")
    
    # Launch the jobs
    job_ids = []
    for worker in range(config["slurm"]["jobs"]):
        w = subprocess.Popen(
            f"sbatch {script_location}", 
            shell=True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE
        ).communicate()
        job_ids.append(
            w[0].strip("\n").split(" ")[-1]
        )
        
    # Write the job ids to file for reference
    with open(os.path.join(script_path, "job_ids.txt"), "w") as fid:
        for line in job_ids:
            fid.write(f"{line}\n")