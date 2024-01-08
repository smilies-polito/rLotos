# Reinforcement Learning based OpTimization Of biofabrication Systems (rLotos)

This work proposes a Deep Reinforcement Learning-based computational Design Space Exploration methodology to generate optimal protocols for the simulated fabrication of epithelial sheets. The optimization strategy relies on a variant of the Advantage-Actor-Critic (A2C) algorithm. Simulations rely on the `PalaCell2D` simulation framework by 

* Conradin, R., Coreixas, C., Latt, J., & Chopard, B. (2021). PalaCell2D: A framework for detailed tissue morphogenesis. Journal of Computational Science, 53, 101353. https://doi.org/10.1016/j.jocs.2021.101353

* Conradin, R. E. Y. (2022). Cell-based simulation of tissue morphogenesis (Doctoral dissertation, University of Geneva). https://archive-ouverte.unige.ch/unige:166037 

## Release notes

rLotos v1.0: 

* First release of rLotos.


## How to cite

* Castrignanò, A., Bardini, R., Savino, A., Di Carlo, S. (2024) A methodology combining reinforcement learning and simulation to optimize biofabrication protocols applied to the simulated culture of epithelial sheets. (Submitted for review to JOCS journal). URL: https://www.biorxiv.org/content/10.1101/2023.04.25.538212v3


* Castrignanò, A. (2022). A reinforcement learning approach to the computational generation of biofabrication protocols (Master Thesis dissertation, Politecnico di Torino). URL: http://webthesis.biblio.polito.it/id/eprint/25391 


## Experimental setup

Follow these steps to set up the optimization engine and reproduce the experiments provided in _Castrignanò et al., 2023_.

1) Install `Singularity` from https://docs.sylabs.io/guides/3.0/user-guide/installation.html:
	* Install `Singularity` release 3.10.2, with `Go` version 1.18.4
	* Suggestion: follow instructions provided in _Download and install singularity from a release_ section after installing `Go`
	* Install dependencies from: https://docs.sylabs.io/guides/main/admin-guide/installation.html

2) Clone the rLotos repository in your home folder

```
git clone https://github.com/smilies-polito/rLotos.git
```

3) Move to the rLotos source subfolder, and build the singularity container with 

```
cd rLotos/source
sudo singularity build rLotos.sif rLotos.def
```
or

```
cd rLotos/source
singularity build --fakeroot rLotos.sif rLotos.def
```

4) Set execution privileges to `palaCell` binary file in `/data` folder

```
cd rLotos/data/app
chmod +x palaCell
```

Notes:

* a processor with 'AVX/AVX2' instruction sets enabled is required.
* if using a virtual machine with Windows, it may be necessary to disable Hyper-V and hypervisor, together with Windows 'device guard' and 'memory integrity' features, as they will make virtual machines unable to use 'AVX/AVX2' instructions from the processor.


## Reproducing the experiments

In _Castrignanò et al., 2023_, provided experiments organize around Target 1 (maximal number of cells) and Target 2 (circular patch of cells), and explores both learning hyperparameters (`lr` and `gamma`) and the `iters` parameter. The rLotos repository follows the following naming rules for experiments based on the proposed Reinforcement Learning (RL) approach or on the Genetic Algorithm (GA) used for comparison:

* **Experiment 1.1**: refers to Target 1 - RL - exploration of different values of learning rate `lr` and `gamma`
* **Experiment 1.2**: refers to Target 1 - RL - exploration of different values of `iters` with fixed values of `lr` and `gamma`
* **Experiment 2.1**: refers to Target 2 - RL - exploration of different values of learning rate `lr` and `gamma`
* **Experiment 2.2**: refers to Target 2 - RL - exploration of different values of `iters` with fixed values of `lr` and `gamma`
* **Experiment 4**: refers to Target 2 - GA - exploration of different values of `iters`

**Experiment 4** relies on the implementation of the `PyGAD` library, an open-source Python 3 library for implementing the genetic algorithm (URL: https://pypi.org/project/pygad/). 

### Reproducing the experiments using the rLotos Singularity container

To reproduce the experiments from _Castrignanò et al., 2023_, run the `rLotos.sif` container with experiment-specific commandline arguments.

**Experiment 1.1**: Target 1 - RL - exploration of different values of learning rate `lr` and `gamma`

```
singularity run --no-home --bind /local/path/to/rLotos:/local/path/to/home/ rLotos.sif experiment1 train_manager.py
```

**Experiment 1.2**: refers to Target 1 - RL - exploration of different values of `iters` with fixed values of `lr` and `gamma`

```
singularity run --no-home --bind /local/path/to/rLotos:/local/path/to/home/ rLotos.sif experiment1 train_manager_iters.py
```

**Experiment 2.1**: refers to Target 2 - RL - exploration of different values of learning rate `lr` and `gamma`

```
singularity run --no-home --bind /local/path/to/rLotos:/local/path/to/home/ rLotos.sif experiment2 train_manager.py

```
**Experiment 2.2**: refers to Target 2 - RL - exploration of different values of `iters` with fixed values of `lr` and `gamma`

```
singularity run --no-home --bind /local/path/to/rLotos:/local/path/to/home/ rLotos.sif experiment2 train_manager_iters.py
```

**Experiment 4**: refers to Target 2 - GA - exploration of different values of `iters`

```
singularity run --no-home --bind /local/path/to/rLotos:/local/path/to/home/ rLotos.sif experiment4 evolve_manager.py
```

### Reproducing the experiments manually

Experiments have default values of `lr` and `gamma`, but hyperparameters can be easily changed in:

* `/source/experiment1/train_manager.py`, for Experiment 1.1
* `/source/experiment1/train_manager_iters.py`, for Experiment 1.2
* `/source/experiment2/train_manager.py`, for Experiment 2.1
* `/source/experiment2/train_manager_iters.py`, for Experiment 2.2
* `/source/experiment4/evolve_manager.py`, for Experiment 4


To run an experiment manually:

**Experiment 1.1**: Target 1 - RL - exploration of different values of learning rate `lr` and `gamma`

```
cd /source/experiment1
pyhon3 train_manager.py
```

**Experiment 1.2**: refers to Target 1 - RL - exploration of different values of `iters` with fixed values of `lr` and `gamma`

```
cd /source/experiment1
pyhon3 train_manager_iters.py
```

**Experiment 2.1**: refers to Target 2 - RL - exploration of different values of learning rate `lr` and `gamma`

```
cd /source/experiment2
pyhon3 train_manager.py
```

**Experiment 2.2**: refers to Target 2 - RL - exploration of different values of `iters` with fixed values of `lr` and `gamma`

```
cd /source/experiment2
pyhon3 train_manager_iters.py
```

**Experiment 4**: refers to Target 2 - GA - exploration of different values of `iters`

```
cd /source/experiment4
pyhon3 evolve_manager.py
```

Note: if `cuda` or `tensorflow` libraries give errors when using the gpu, it it possible to switch to CPU usage only by uncommenting the following line: `#tf.config.set_visible_devices([], 'GPU')`.
This line is present in:
	* `source/experiment1/train.py`
	* `source/experiment2/train.py`
	* `source/experiment2/palacellProlifTrain.py`


## Handling output files

The proposed experiments' output files are organized as `Python` `dicts`, containing lists of useful data for both starting the training from a previous checkpoint, and for reading the results of the training.
Output files names refer to the specific epoch at which they have been generated.

### Restoring the training from a checkpoint

* Open the experiment's train manager python file:
	* `/source/experiment1/train_manager.py` for Experiment 1.1
	* `/source/experiment1/train_manager_iters.py` for Experiment 1.2
	* `/source/experiment2/train_manager.py`, for Experiment 2.1
	* `/source/experiment2/train_manager_iters.py`, for Experiment 2.2
* Set the variable `starting_epoch` to the desired epoch (for example, the last available epoch)
* Follow the commented instructions below the environment creation in the file
* Note: check the 'preload' uses the correct file path
* Note: for the training to give coherent results, all loaded checkpoint files must refer to the same starting epoch

### Reading the RL training output

* Load the files with `numpy.load` (make sure to use `allow_pickle=True`)

Output files contain `Python` `dicts` with the following structure:

* `data_to_save_at_epoch_X.npy` contains the training results 
	* for Experiments 1.1 and 1.2:
		* `cell_numbers` contains the number of cells at the end of each epoch
		* `cell_increments` contains the difference in the number of cells between successive epochs
		* `compr_history` contains the list of actions for every simulation iteration at each epoch
	* for Experiment 2:
		* for the main training process:
			* `inside_outside` contains tuples with the fraction of cells inside and outside the target area, with respect to the total number of cells, for each epoch until epoch X
			* `circle_actions` contains the list of actions for every simulation iteration at each epoch
		* for the inner training process:
			* `cell_numbers` contains the number of cells at the end of each epoch
			* `cell_increments` contains the difference in the number of cells between successive epochs
			* `compr_history` contains the list of actions for every simulation iteration at each epoch
* `model_at_epoch_X.npy` allows to start the training from a previous checkpoint (at epoch X)
* `performance_at_epoch_X.npy` contains performance indices needed by the environment at epoch X
	* for Experiments 1.1 and 1.2:
		* `best_cell_num` contains the maximum number of cells obtained for each epoch until epoch X
	* for Experiment 2:
		* for the main training process:
			* `best_cell_num` contains the tuple referring to the maximum fraction of cells inside the target area, with respect to the total number of cells, until epoch X
		* for the inner training process:
			* `best_cell_num` contains the maximum number of cells obtained for each epoch until epoch X

An example of how to read the training results can be found in the `read_output.py` files in each experiment folder.

### Launching the RL in testing mode

In order to launch a set of testing epochs for any RL experiment, it is necessary to set the `testingMode` parameter to `True` when launching the training: 

```
parallel_train(testingMode=True)
```
Otherwise, the option defaults to `False`.

## Repository structure

```
|
├── data                                        // Data files
|   └── PalaCell2D                              // Data files supporting PalaCell2D
|       ├── app                                 // Simulator files
|       |    ├── palaCell                       // Executable of the simulator used in the provided experiments
|       |    ├── output                         // Created during the experiments, contains data files used by the simulator
|       |    └── ...                            // Configuration files used by the simulator, created during the experiments
|       └── external                            // Libraries used by palaCell program
│
│
├── source                                   // Source code of the project
│   ├── experiment1                          // Source files of experiment 1, both experiment1.1 and experiment1.2
|   |   ├── env                              // Environment of the RL algorithm for experiment 1
|   |   |   ├── checks.py                    // Checks to be made on provided environment for experiment 1
|   |   |   ├── PalacellEnv.py               // Interface used by the optimization engine to control the simulations for experiment 1
|   |   |   └── vtkInterface.py              // Useful functions to use the vtk library for experiment 1
|   |   ├── model.py                         // Neural network definition for experiment 1
|   |   ├── read_output.py                   // Reads output files for experiment 1.1
|   |   ├── read_output_iters.py             // Reads output files for experiment 1.2
|   |   ├── train_manager_iters.py           // Starts the training of experiment 1.2
|   |   ├── train_manager.py                 // Starts the training of experiment 1.1
|   |   └── train.py                         // Performs the training for experiment 1
│   ├── experiment2                          // Source files of experiment 2
|   |   ├── env                              // Environment of the RL algorithm for experiment 2
|   |   |   ├── checks.py                    // Checks to be made on provided environment for experiment 2
|   |   |   ├── PalacellEnv.py               // Interface used by the optimization engine to control the simulations for experiment 1
|   |   |   └── vtkInterface.py              // Useful functions to use the vtk library for experiment 2
|   |   ├── model.py                         // Neural network definition for experiment 2
|   |   ├── palacellProlifTrain.py           // Helper used by the environment to execute the inner proliferation part of the training for experiment 2
|   |   ├── read_output.py                   // Reads output files for experiment 2
|   |   ├── train_manager.py                 // Starts the training of experiment 2
|   |   └── train.py                         // Performs the training for experiment 2
│   ├── experiment4                          // Source files of experiment 4
|   |   ├── env                              // Environment of the RL algorithm for experiment 2
|   |   |   ├── checks.py                    // Checks to be made on provided environment for experiment 2
|   |   |   ├── PalacellEnv.py               // Interface used by the optimization engine to control the simulations for experiment 1
|   |   |   └── vtkInterface.py              // Useful functions to use the vtk library for experiment 2
|   |   ├── evolve_manager.py                // Starts the evolutionary process of experiment 4
|   |   └── evolve.py                        // Performs the evolution for experiment 4
|   ├── EnvironmentBlueprint.py              // Template to build an environment
|   ├── generate_plots.py                    // Script to generate figures from RL and GA outputs
|   └── rLotos.def                           // Singularity recipe
|
└── README.md                                // This README file          
```

