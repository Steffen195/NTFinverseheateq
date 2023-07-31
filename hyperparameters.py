from model.network import Sine, AdaptiveSoftPlus, AdaptiveTanh
import torch.nn as nn

hyperparameter_dict = {'lbfgs_learning_rate': 1, 'adam_learning_rate': 0.0001, 'adam_epochs': 20, 'lbfgs_epochs': 50,'lbfgs_history_size': 10,
                        'step_size_LBFGS': 50,"step_size_adam": 2000,
                        'weight_physical': 1.5, 'weight_data': 1, 'weight_boundary': 1, 'weight_initial': 1, 'weight_sparsity': 0,
                        'batch_size_physical': 50, "n_collocation_points":10000,
                        'width': 275, 
                        'output_activation': nn.Identity(),'activation': nn.Tanh(),
                        'num_layers_source': 5, "num_layers_temperature": 7,
                        'spatial_gridpoints': 200, 'temporal_gridpoints': 1000, 'sensor_interval': 40, 'data_loader_length': 125}




