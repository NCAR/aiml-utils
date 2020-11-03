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
    
    # Set up new db entry if reload = 0 
    reload_study = bool(hyper_config["optuna"]["reload"])
        
    if not reload_study:        
        name = hyper_config["optuna"]["name"]
        storage =f"sqlite:///{name}"
        
        if os.path.isfile(name):
            os.remove(name)
        
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