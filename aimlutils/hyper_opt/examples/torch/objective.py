import warnings
warnings.filterwarnings("ignore")

import copy
import optuna
import logging
import traceback

from overrides import overrides
from holodecml.vae.losses import *
from holodecml.vae.visual import *
from holodecml.vae.models import *
from holodecml.vae.trainers import *
from holodecml.vae.transforms import *
from holodecml.vae.optimizers import *
from holodecml.vae.data_loader import *
from holodecml.vae.checkpointer import *
from aimlutils.hyper_opt.base_objective import *

from torch import nn
from torch.optim.lr_scheduler import *
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple


logger = logging.getLogger(__name__)


def custom_updates(trial, conf):
    
    # Get list of hyperparameters from the config
    hyperparameters = conf["optuna"]["parameters"]
    
    # Now update some via custom rules
    num_dense = trial_suggest_loader(trial, hyperparameters["num_dense"]) 
    dense1 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim1'])
    dense2 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim2'])
    dr1 = trial_suggest_loader(trial, hyperparameters['dr1'])
    dr2 = trial_suggest_loader(trial, hyperparameters['dr2'])
    
    # Update the config based on optuna suggestions
    conf["model"]["dense_hidden_dims"] = [dense1] + [dense2 for k in range(num_dense)]        
    conf["model"]["dense_dropouts"] = [dr1] + [dr2 for k in range(num_dense)]
    return conf     


class Objective(BaseObjective):
    
    def __init__(self, study, config, metric = "val_loss", device = "cpu"):
        
        BaseObjective.__init__(self, study, config, metric, device)


    def train(self, trial, conf):   
        
        ###########################################################
        #
        # Implement custom changes to config
        #
        ###########################################################
        
        conf = custom_updates(trial, conf)
                
        ###########################################################
        #
        # Load ML pipeline, train the model, and return the result
        #
        ###########################################################
        
        results = {
            "val_loss": val_loss, 
            "val_mse": val_mse, 
            "val_bce": val_bce, 
            "val_acc": val_acc
        }
        
        return self.save(trial, results)


class CustomTrainer(BaseEncoderTrainer):

    def train(self,
              trial,
              scheduler,
              early_stopping, 
              metric = "val_loss"):

        flag = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        for epoch in range(self.start_epoch, self.epochs):
            
            try:
                train_loss, train_mse, train_bce, train_accuracy = self.train_one_epoch(epoch)
                test_loss, test_mse, test_bce, test_accuracy = self.test(epoch)
            
                if "val_loss" in metric:
                    metric_val = test_loss
                elif "val_mse_loss" in metric:
                    metric_val = test_mse
                elif "val_bce_loss" in metric:
                    metric_val = test_bce
                elif "val_acc" in metric:
                    metric_val = -test_accuracy
                else:
                    supported = "val_loss, val_mse_loss, val_bce_loss, val_acc"
                    raise ValueError(f"The metric {metric} is not supported. Choose from {supported}")

                trial.report(-metric_val, step=epoch+1)
                scheduler.step(metric_val if flag else epoch)
                early_stopping(epoch, metric_val, self.model, self.optimizer)
                
            except Exception as E: # CUDA memory overflow
                print(traceback.print_exc())
                raise optuna.TrialPruned()
            
            if trial.should_prune():
                raise optuna.TrialPruned()
                
            if early_stopping.early_stop:
                break
                
        return test_loss, test_mse, test_bce, test_accuracy