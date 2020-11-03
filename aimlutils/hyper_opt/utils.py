import optuna


supported_trials = [
    "categorical",
    "discrete_uniform",
    "float",
    "int",
    "loguniform",
    "uniform"
]

def trial_suggest_loader(trial, config):
    _type = config["type"]
    if _type not in supported_trials:
        raise OSError(
            f"Type {_type} is not valid. Select from {supported_trials}"
        )
    if _type == "categorical":
        return trial.suggest_categorical(**config["settings"])
    elif _type == "discrete_uniform":
        return trial.suggest_discrete_uniform(**config["settings"])
    elif _type == "float":
        return trial.suggest_float(**config["settings"])
    elif _type == "int":
        return trial.suggest_int(**config["settings"])
    elif _type == "loguniform":
        return trial.suggest_loguniform(**config["settings"])
    elif _type == "uniform":
        return trial.suggest_uniform(**config["settings"])
    

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
        raise OSError(
            f"Sampler {_type} is not valid. Select from {supported_samplers}"
        )
    if _type == "TPESampler":
        return optuna.samplers.TPESampler()
    elif _type == "GridSampler":
        if "search_space" in sampler:
            return optuna.samplers.GridSampler(sampler["search_space"])
        else:
            return optuna.samplers.GridSampler()
    elif _type == "RandomSampler":
        return optuna.samplers.RandomSampler()
    elif _type == "CmaEsSampler":
        return optuna.integration.CmaEsSampler()
    elif _type == "IntersectionSearchSpace":
        return optuna.integration.IntersectionSearchSpace()