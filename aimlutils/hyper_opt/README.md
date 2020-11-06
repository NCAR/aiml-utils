# hyper_opt: A distributed multi-gpu hyperparameter optimization package build with optuna

### Usage

python optimize.py hyperparameters.yml model_config.yml

### Dependencies

There are three files that must be supplied to use the optimize script:

* A custom objective class that trains your model and returns the metric to be optimized.

* A configuration file specifying the hyperparameter optimization settings.

* A model configuration file that contains the information needed to train your model (see examples in the holodec and gecko projects).

### Custom objective class

The custom **Objective** class (objective.py) must be composed with a **BaseObjective** class (which lives in base_objective.py), and must contain a method named **train** that returns the value of the optimization metric (in a dictionary, see below). There are example objective scripts for both torch and Keras in the examples directory. Your custom Objective class will inherit all of the methods and attributes from the BaseObjective (this way of doing of OOP is called composition). The Objective's train does not depend on the machine learning library used! For example, a simple template has the following structure:

    from aimlutils.hyper_opt.base_objective import *
    from aimlutils.hyper_opt.utils import KerasPruningCallback

    class Objective(BaseObjective):

        def __init__(self, study, config, metric = "val_loss", device = "cpu"):

            # Initialize the base class
            BaseObjective.__init__(self, study, config, metric, device)

        def train(self, trial, conf):

            # Make any custom edits to the model conf before using it to train a model.
            conf = custom_updates(trial, conf)

            ... 
            
            callbacks = [KerasPruningCallback(trial, self.metric, interval = 1)]
            result = Model.fit(..., callbacks = callbacks)
            
            if trial.should_prune():
                raise optuna.TrialPruned()

            results_dictionary = {
                "val_loss": result["val_loss"],
                "loss": result["loss"],
                ...
                "val_accuracy": result["val_accuracy"]
            }
            return results_dictionary

You can have as many inputs to your custom Objective as needed, as long as those that are required to initialize the base class are included. The Objective class will call the train method from the inherited thunder **__call__** method, and will finish up by calling the inherited save method that writes the metric(s) details to disk. Note that, due to the composition of the two classes, you do not have to supply these two methods, as they are in pre-coded in the base class! You can customize them at your leisure using overriding methods in your custom Objective. Check out the scripts base_objective.py and run.py to see how things are composed and called.

As noted, the metric used to toggle the model's training performance must be in the results dictionary. Other metrics that the user may want to track will be saved to disk if they are included in the results dictionary (the keys of the dictionary are used to name the columns in a pandas dataframe). See the example above where several metrics are being returned.

Note that the first line in the train method states that any custom changes to the model configuration (conf) must be done here. If custom changes are required, the user may supply a method named **custom_updates** in addition to the Objective class (you may save both in the same script, or import the method from somewhere else in your custom Objective script). See also the section **Custom model configuration updates** below for an example. 

Finally, if using Keras, you need to include the KerasPruningCallback that will allow optuna to termine unpromising trials. We do something similar when using torch -- see the examples directory.

### Hyperparameter optimizer configuration

There are three main fields, log, slurm, and optuna, and variable subfields within each field. The log field allows us to save a file for printing messages and warnings that are placed in areas throughout the package. The slurm field allows the user to specify how many GPU nodes should be used, and supports any slurm setting. The optuna field allows the user to configure the optimization procedure, including specifying which parameters will be used, as well as the performance metric. For example, consider the configuration settings:

* log
  + save_path: "path/to/data/log.txt"
* slurm
  + jobs: 20
  + batch:
    + account: "NAML0001"
    + gres: "gpu:v100:1"
    + mem: "128G"
    + n: 8
    + t: "12:00:00"
    + J: "hyper_opt"
    + o: "hyper_opt.out"
    + e: "hyper_opt.err"
* optuna
  + name: "holodec_optimization.db"
  + reload: 0
  + objective: "examples/torch/objective.py"
  + metric: "val_loss"
  + direction: "minimize"
  + n_trials: 500
  + gpu: True
  + save_path: 'test'
  + sampler:
    + type: "TPESampler"
  + parameters:
    + num_dense:
      + type: "int"
      + settings:
        + name: "num_dense"
        + low: 0
        + high: 10
    + dropout:
      + type: "float"
      + settings:
        + name: "dr"
        + low: 0.0
        + high: 0.5
    + **optimizer:learning_rate**:
      + type: "loguniform"
      + settings:
        + name: "lr"
        + low: 0.0000001
        + high: 0.01

The subfields within the optuna field have the following functionality:

* name: The name of the study.
* reload: Whether to continue using a previous study (True) or to initialize a new study (False). If your initial number of workers do not reach the number of trials and you wish to resubmit, set to True.
* objective: The path to the user-supplied objective class (it must be named objective.py)
* metric: The metric to be used to determine the model performance. 
* direction: Indicates which direction the metric must go to represent improvement (pick from maximimize or minimize)
* n_trials: The number of trials in the study.
* gpu: Use the gpu or cpu.
* save_path: Directory path where data will be saved. 
* sampler
  + type: Choose how optuna will do parameter estimation. The default choice both here and in optuna is the [Tree-structured Parzen Estimator Approach](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f), [e.g. TPESampler](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf). See the optuna documentation for the different options. For some samplers (e.g. GridSearch) additional fields may be included (e.g. search_space). 
* parameters
  + type: Option to select an optuna trial setting. See the [optuna Trial documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html?highlight=suggest#optuna.trial.Trial.suggest_uniform) for what is available. Currently, this package supports the available options from optuna: "categorical", "discrete_uniform", "float", "int", "loguniform", and "uniform".
  + settings: This dictionary field allows you to specify any settings that accompany the optuna trial type. In the example above, the named num_dense parameter is stated to be an integer with values ranging from 0 to 10. To see all the available options, consolt the [optuna Trial documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html?highlight=suggest#optuna.trial.Trial.suggest_uniform)

### Model configuration

The model configuration file can be what you had been using up to this point to train your model, in other words no changes are necessary. This package will take the suggested hyperparameters from an optuna trial and make changes to the model configuration on the fly. This can either be done automatically with this package, or the user may supply an additional method for making custom changes. For example, consider the (truncated) configuration for training a model to predict hologram properties with a holodec dataset:

* model:
  + image_channels: 1
  + hidden_dims: [3, 94, 141, 471, 425, 1122]
  + z_dim: 1277
  + dense_hidden_dims: [1000]
  + dense_dropouts: [0.0]
  + tasks: ["x", "y", "z", "d", "binary"]
+ **optimizer**:
  * type: "lookahead-diffgrad"
  * **learning_rate**: 0.000631
  * weight_decay: 0.0
+ trainer:
  * start_epoch: 0
  * epochs: 1
  * clip: 1.0
  * alpha: 1.0
  * beta: 0.1
  * path_save: "test"
  
The model configuration will be automatically updated if and only if the name of the parameter specified in the hyperparameter configuration, optuna.parameters can be used as a nested lookup key in the model configuration file. For example, observe in the hyperparameter configuration file above that the named parameter **optimizer:learning_rate** contains a colon, that is downstream used to split the named parameter into multiple keys that allow us to, starting from the top of the nested tree in the model configuration file, work our way down until the relevant field is located and the trial-suggested value is substituted in. In this example, the split keys are ["optimizer", "learning_rate"]. 

This scheme will work in general as long as the named parameter in optuna.parameters uses : as the separator, and once split, the resulting list can be used to locate the relevant field in the model configuration.


### Custom model configuration updates

You may additionally supply rules for updating the model configuration file, by including a method named **custom_updates**, which will make the desired changes to the configuration file with optuna trail parameter guesses.

In the example configurations described above, the hyperparameter configuration contained an optuna.parameters field "num_dense," but this field is not present in the model configuration. There is however a "dense_hiddden_dims" field in the model configuration that contains a list of the layer sizes in the model, where the number of layers is the length of the list. In our example just one layer specified but we want to vary that number. To use the "num_dense" hyperparameter from the hyperparameter configuration file, we can create the following method:

    def custom_updates(trial, conf):
    
        # Get list of hyperparameters from the config
        hyperparameters = conf["optuna"]["parameters"]
    
        # Now update some via custom rules
        num_dense = trial.suggest_discrete_uniform(**hyperparameters["num_dense"]) 
    
        # Update the config based on optuna's suggestion
        conf["model"]["dense_hidden_dims"] = [1000 for k in range(num_dense)]        
        
        return conf 
        
The method should be called first thing in the custom Objective.train method (see the example Objective above). You may have noticed that the configuration (named conf) contains both hyperparameter and model fields. This package will copy the hyperparameter optuna field to the model configuration for convenience, so that we can reduce the total number of class and method dependencies (which helps me keep the code generalized). This occurs in the run.py script.

One final remark about the types of trial parameters optuna will support, which were noted a few passages above. In short, optuna has a limited range of the types of trial parameters, all of them being numerical in one form or another (float or int). If you wanted to optimize the activation layer(s) in your neural network, you could go about that by utilizing the "categorical" trial suggestor and proceeding to "tokenize" a list of potential activations. For example, we could "tokenize" the following list of activation layers ["relu", "linear", "leaky-relu", "tanh", "sigmoid"] by simply assigning them integers from a categorical trial suggestor as follows: [0, 1, 2, 3, 4]. Optuna will then attempt to optimize the tokenized list (rather than the explicit activation names), eventually settling on one of them (lets say its 2). Then its just a matter of doing a reverse-lookup to find that "leaky-relu" was the best performing activation layer.