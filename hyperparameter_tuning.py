import random
from math import log10
import torch.nn as nn
import logging
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model.network import TemperatureNet, SourceNet, Sine, HeatNet
from temperature_sim import Grid, Source, HeatEqSimulation
from prediction_evaluation import evaluate_model
from training import train_model, create_full_domain_input_data
from hyperparameters import hyperparameter_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using the following device: ', device)

ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']

def random_search(random_search_space, num_search):
    """
    Creates list of hparam configs for hyperparameter tuning
    """
    configs = []

    for _ in range(num_search):
        configs.append(random_search_space_to_config(random_search_space))

    return configs

def random_search_space_to_config(random_search_space):
    config = {}

    for key, (range, mode) in random_search_space.items():
            
            if mode not in ALLOWED_RANDOM_SEARCH_PARAMS:
                raise ValueError('Invalid mode: {}'.format(mode))      
            elif mode == 'log':
                config[key] = random.uniform(log10(range[0]), log10(range[1]))
                config[key] = 10 ** config[key]
            elif mode == 'int':
                config[key] = random.randint(range[0], range[-1])
            elif mode == 'float':
                config[key] = random.uniform(range[0], range[-1])
            elif mode == 'item':
                config[key] = random.choice(range)

    return config

def main():
    """
    Conducts hyperparameter tuning for the heat equation
    """
    logging.basicConfig(filename='hyperparameter_tuning.log', level=logging.INFO)
    logging.info('Started Hyperparameter Tuning')
    hparam = hyperparameter_dict
    search_space = {"weights_pyhsical":([0.5,2],"float"),
                    "weights_data":([0.5,2],"float"),
                    "weights_boundary":([0.5,2],"float"),
                    "weights_initial":([0.5,2],"float"),}
                    
    configs = random_search(search_space, 100 )
    

    grid = Grid(spatial_gridpoints=hparam["spatial_gridpoints"],temporal_gridpoints=hparam["temporal_gridpoints"])
    
    parabola_function = lambda t,x : max([0.0,-3*x**2+0.3])
    gaussian_function =  lambda t,x : np.exp(-1/2 * ((x**2)/0.1)) * 1/np.sqrt(2*np.pi*0.1**2)
    moving_gaussian_function =  lambda t,x : np.exp(-1/2 * (((x-(t-1)))**2)/0.1) * 1/np.sqrt(2*np.pi*0.1**2)

    best_source_error = np.inf
    best_source_config = None
    best_temperature_error = np.inf
    best_temperature_config = None

    source_function_list = [parabola_function,moving_gaussian_function,gaussian_function]
    function_weight_list = [0.2,0.3,0.5]
    
    for index, config in enumerate(configs):
        
        logging.info(f'====== Started  training {index} with config: {config}=====')
        hparam.update(config)
        logging.info('===Hyperparameter: {}==='.format(hparam))

        total_source_mse = 0
        total_temperature_mse = 0
        total_source_error = 0
        total_temperature_error = 0

        try:
            for index, function in enumerate(source_function_list):
                source = Source.from_function(grid,function)
                sim = HeatEqSimulation(source, grid,sensor_interval= hparam["sensor_interval"])

                model, losses = train_model(sim, hparam,verbose = False)
                    
                #model evaluation
                summary = evaluate_model(sim,model, losses= losses, device=device,plot = False)
                
                total_source_mse += summary["source_mean_square_error"]*function_weight_list[index]
                total_temperature_mse += summary["temperature_mean_square_error"]*function_weight_list[index]
                total_source_error += summary["source_total_error"]*function_weight_list[index]
                total_temperature_error += summary["temperature_total_error"]*function_weight_list[index]

        except Exception as e:
                logging.error('Training with Config {} failed with error: {}\n\n'.format(config, e))
                continue
        
        logging.info('Finished Training with Config: {}'.format(config))
        logging.info('Summary: {}\n\n'.format(summary))
        logging.info('Total Source MSE: {}'.format(total_source_mse))
        logging.info('Total Temperature MSE: {}'.format(total_temperature_mse))
        logging.info('Total Source Error: {}'.format(total_source_error))
        logging.info('Total Temperature Error: {}'.format(total_temperature_error))
        

        if total_source_error < best_source_error:
            best_source_error = total_source_error
            best_source_config = config
            torch.save(model.state_dict(), 'trained_models/best_source_model.pt')
            logging.info('New best source model.\n')

        if total_temperature_error < best_temperature_error:
            best_temperature_error = total_temperature_error
            best_temperature_config = config
            torch.save(model.state_dict(), 'trained_models/best_temperature_model.pt')
            logging.info('New best temperature model.\n')

    logging.info('Finished Hyperparameter Tuning')
    logging.info('Best Source Model: {}'.format(best_source_config))
    logging.info('Best Temperature Model: {}'.format(best_temperature_config))
    logging.info('Best Source Error: {}'.format(best_source_error))
    logging.info('Best Temperature Error: {}'.format(best_temperature_error))




if(__name__ == "__main__"):
    main()