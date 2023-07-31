# NTF Project - Solution of inverse problem: Heat Equation with Source
- We simulate the 1D heat equation with source in temperature_sim.py
- This simulation data is used to train a physics informed neural network. The architecture is found in model/network.py. The training is performed in  training.py using the hyperparameters from hyperparameters.py
- The training can be observed using tensorboard. The logs are stored in the logs directory.
- The performance is evaulated in prediction_evaluation.py, containg functions for plotting and calculating the error
- utils.py contains functions for manipulating and reshaping data.
- Hyperparameter tuning with random search is implemented in hyperparameter_tuning.py
