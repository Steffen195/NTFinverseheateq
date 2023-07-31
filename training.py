import os
import numpy as np
import torch
from torch.autograd import grad
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from model.network import TemperatureNet, SourceNet, Sine, HeatNet
from temperature_sim import Grid, Source, HeatEqSimulation
from prediction_evaluation import evaluate_model
from utils import create_full_domain_input_data, create_sensor_data, create_collocation_points
from hyperparameters import hyperparameter_dict

#Setup the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using the following device: ', device)

# setting a seed for pytorch as well as one for numpy
torch.manual_seed(2)
np.random.seed(2)

#select hyperparameters from hyperparameters.py
hparam = hyperparameter_dict

def loss_function(T_pred, output, T_t, T_xx, source_strength_pred, hparam,T_0,T_left,T_right,T_NN_0,T_NN_left,T_NN_right,sim,
                  epoch,tb_logger,batch_idx=0):
    """
    This function defines the loss of the PINN. The Loss consists of a physical loss, a data loss, an initial loss and a boundary loss.

    """

    # The physical loss is calculated at the collocation points given by hparams["n_collocation_points"].
    # The loss is equivalen to the residual of the heat equation. All derivates are calculated using automatic differentiation outside of the loss funciton. 

    physical_loss = hparam["weight_physical"]*torch.mean((T_t - sim.conductivity*T_xx - source_strength_pred)**2)

    # Data Loss
    data_loss = hparam["weight_data"]*torch.mean((T_pred - output[:,0])**2)

    # Initial and Boundary Loss
    initial_loss = hparam["weight_initial"]*torch.mean((T_0 - T_NN_0)**2)
    boundary_loss = hparam["weight_boundary"]*torch.mean(((T_left - T_NN_left) + (T_right-T_NN_right))**2)

    # The current learning step is calculated differently depending on whether the model is in the Adam or LBFGS phase
    if epoch < hparam["adam_epochs"]:
        current_learning_step = epoch * hparam["data_loader_length"] + batch_idx
    else:
        current_learning_step = (hparam["adam_epochs"]+1) * hparam["data_loader_length"] + (epoch) - hparam["adam_epochs"]

    # Logging of the losses to tensorboard
    tb_logger.add_scalar('Loss/Physical', physical_loss, current_learning_step)
    tb_logger.add_scalar('Loss/Data', data_loss, current_learning_step)
    tb_logger.add_scalar('Loss/Initial', initial_loss, current_learning_step)
    tb_logger.add_scalar('Loss/Boundary', boundary_loss, current_learning_step) 


    loss = data_loss + physical_loss + boundary_loss + initial_loss

    tb_logger.add_scalar('train_loss', loss.item(), current_learning_step)
    
    return loss


def train_model(sim :HeatEqSimulation,hparam,verbose=False):

    # setting up the tensorboard logger
    path = os.path.join('logs', 'SegPinn')
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path = os.path.join(path, f'run_{num_of_runs + 1}')
    tb_logger = SummaryWriter(path)

    
    # Get the data and reshape it into the correct format. 
    sensor_input, sensor_output = create_sensor_data(sim)
    sensor_input.requires_grad = True
    sensor_input = sensor_input.to(device)
    sensor_output = sensor_output.to(device)

    # The full domain input containes the collection of all collocation points, where the physical loss is calculated.
    full_domain_input = create_collocation_points(hparam["n_collocation_points"],sim.grid)
    full_domain_input.requires_grad = True
    full_domain_input = full_domain_input.to(device)


    # The simulation domain_input is the coordinates, in which the simulation is performed and differs from the full domain input.
    simulation_domain_input = create_full_domain_input_data(sim.grid).to(device)

    # Create torch dataset from the full_domain_input data with no output
    test_dataset = TensorDataset(full_domain_input)
    test_loader = DataLoader(test_dataset, batch_size=hparam["batch_size_physical"], shuffle=True,drop_last=True)
    hparam["data_loader_length"] = len(test_loader)

    losses = np.zeros(hparam["adam_epochs"]+hparam["lbfgs_epochs"])

    # Instantiate the model, optimizer, data loaders and loss function
    heatNet = HeatNet(hparam).to(device)
    optimizer = Adam(heatNet.parameters(), lr=hparam["adam_learning_rate"])
    scheduler = StepLR(optimizer, step_size=hparam["step_size_adam"], gamma=0.5)

    # The ground truth temperature at the boundaries and initial condition is calculated from the simulation domain input
    index_t0 = simulation_domain_input[:,0] == sim.grid.t[0]
    index_left = simulation_domain_input[:,1] == sim.grid.x[0]
    index_right = simulation_domain_input[:,1] == sim.grid.x[-1]

    input_bc_left = simulation_domain_input[index_left]
    input_bc_right = simulation_domain_input[index_right]
    input_ic = simulation_domain_input[index_t0]
  
    T_0 = torch.Tensor(sim.temperature_field[0,:]).to(device)
    T_left = torch.Tensor(sim.temperature_field[:,0]).to(device)
    T_right = torch.Tensor(sim.temperature_field[:,-1]).to(device)


    # Training loop
    for epoch in range(hparam["adam_epochs"]):
        training_loss = 0
        # The data loader loads the full domain input data in batches, as the full domain input data is too large to be loaded at once.
        # For every batch of full domain, the complete set of sensor data is loaded.
        for batch_idx, full_domain_input in enumerate(test_loader):
            # Reset the optimizer
            optimizer.zero_grad()

            # Forward pass at the sensor locations 
            T_pred = heatNet(sensor_input).to(device)[:,0]

            # Forward pass at the initial and boundary locations
            T_NN_0 = heatNet(input_ic).to(device)[:,0]
            T_NN_left = heatNet(input_bc_left).to(device)[:,0]
            T_NN_right = heatNet(input_bc_right).to(device)[:,0]

            #Take the data from tensor list as "full domain input" also contain gradient and device information
            full_domain_input =  full_domain_input[0] 
            full_domain_input = full_domain_input.to(device)
            # Complete the forward pass on the full domain
            full_domain_output = heatNet(full_domain_input).to(device)
            
            # Extract the temperature and source strength from the full domain output
            T_pred_physical, source_strength_pred = full_domain_output[:,0],full_domain_output[:,1]
           
            # Automatic differentiation of the temperature field. T_x = dT/dx, T_t = dT/dt, T_xx = d^2T/dx^2
            dT = grad(T_pred_physical.sum(), full_domain_input, retain_graph= True, create_graph=True)[0]
            T_x = dT[:,1]
            T_t = dT[:,0]
            T_xx = grad(T_x, full_domain_input,grad_outputs= torch.ones((hparam["batch_size_physical"])).to(device), retain_graph=True, create_graph=True)[0][:,1]
        
            # Loss calculation
            loss = loss_function(T_pred, sensor_output, T_t, T_xx, source_strength_pred, hparam, T_0, T_left, T_right, T_NN_0, T_NN_left, T_NN_right,sim,epoch,tb_logger,batch_idx)
           
            
            # Backward pass
            loss.backward(retain_graph=True)
            # Update parameters
            optimizer.step()
            scheduler.step()

             # Logging
            training_loss += loss.item()
            
            
        losses[epoch] = training_loss/len(test_loader)

        if verbose:
            print("Epoch: %d, Loss: %.7f" % (epoch, losses[epoch]))


    # Save the model after the Adam phase
    torch.save(heatNet.state_dict(), "trained_models/heatPinn.pt")
    # Clears the cache of the GPU
    torch.cuda.empty_cache()

    #Create new optimizer object for LBFGS, which is a quasi Newton method that brings down the error quite substantially.
    optimizer = LBFGS(heatNet.parameters(),lr = hparam["lbfgs_learning_rate"],history_size=hparam["lbfgs_history_size"],max_iter = 20)
    scheduler = StepLR(optimizer, step_size=hparam["step_size_LBFGS"], gamma=0.1)

    for epoch in range(hparam["adam_epochs"],hparam["adam_epochs"]+hparam["lbfgs_epochs"]):
        training_loss = 0
        
        def closure():
            training_loss = 0
            full_domain_input = create_collocation_points(hparam["n_collocation_points"],sim.grid)
            full_domain_input.requires_grad = True
            full_domain_input = full_domain_input.to(device)
            optimizer.zero_grad()
            # Forward pass
            T_pred = heatNet(sensor_input).to(device)[:,0]
            # Boundary Conditions   
            T_NN_0 = heatNet(input_ic).to(device)[:,0]
            T_NN_left = heatNet(input_bc_left).to(device)[:,0]
            T_NN_right = heatNet(input_bc_right).to(device)[:,0]
            #Take the data from tensor list
            full_domain_input = full_domain_input.to(device)
            full_domain_output = heatNet(full_domain_input).to(device)
            T_pred_physical = full_domain_output[:,0]
            source_strength_pred = full_domain_output[:,1]
        
            # Automatic differentiation
            dT = grad(T_pred_physical.sum(), full_domain_input, retain_graph= True, create_graph=True)[0]
            T_x = dT[:,1]
            T_t = dT[:,0]
            T_xx = grad(T_x, full_domain_input,grad_outputs= torch.ones((len(full_domain_input))).to(device), retain_graph=True, create_graph=True)[0][:,1]
        
            # Loss calculation
            loss = loss_function(T_pred, sensor_output, T_t, T_xx, source_strength_pred, hparam, T_0, T_left, T_right, T_NN_0, T_NN_left, T_NN_right,sim,epoch,tb_logger)
        
            # Backward pass
            loss.backward(retain_graph=True)
            training_loss += loss.item()
            losses[epoch-1] = training_loss
            
            return loss

        if ((losses[epoch-1] == torch.nan)):
            # Terminate, if a singular approximation to the Hessian causes the loss to blow up.
            print("Loss is nan")
            break

        optimizer.step(closure)
        scheduler.step()
    
        if verbose:
            print("Epoch: %d, Loss: %.7f" % (epoch, losses[epoch-1]))


        torch.cuda.empty_cache()
    # Save the model after the LBFGS phase
    torch.save(heatNet.state_dict(), "trained_models/heatPinn.pt")
    
    return heatNet, losses




def main():
    # Create a grid object to define the spatial and temporal grid
    grid = Grid(spatial_gridpoints=hparam["spatial_gridpoints"],temporal_gridpoints = hparam["temporal_gridpoints"])

    # Possible Lambda functions for the source term
    parabola_function = lambda t,x : max([0.0,-3*x**2+0.3])
    gaussian_function =  lambda t,x : np.exp(-1/2 * ((x**2)/0.1)) * 1/np.sqrt(2*np.pi*0.1**2)
    moving_gaussian_function =  lambda t,x : np.exp(-1/2 * (((x-(t-1)))**2)/0.1) * 1/np.sqrt(2*np.pi*0.1**2)

    # Create a source object from the source function
    source = Source.from_function(grid,gaussian_function)
    # Create and run the HeatEq Simulation
    sim = HeatEqSimulation(source, grid,sensor_interval= hparam["sensor_interval"])

    # In case the last trained model should be evaluated again, set preload to True
    preload = False
    
    print(hparam)
    if preload == False:
        #Train the model
        heatNet, losses = train_model(sim,hparam,verbose = True)
        torch.save(heatNet.state_dict(), "trained_models/heatPinn.pt")
        
            
    else:
        # Load the last trained model
        heatNetStateDict = torch.load("trained_models/heatPinn.pt") 
        heatNet = HeatNet(hparam=hparam).to(device)
        heatNet.load_state_dict(heatNetStateDict)
        losses = None

    # Evaluate model and print summary
    summary = evaluate_model(sim,heatNet,losses,device,plot=True)
    print(summary)

if __name__ == "__main__":

    main()