import os
import numpy as np
import torch
from model.network import Net
from torch.autograd import grad
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from temperature_sim import Grid, Source, HeatEqSimulation



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using the following device: ', device)

 # setting up the tensorboard logger
path = os.path.join('logs', 'Case_0')
num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
path = os.path.join(path, f'run_{num_of_runs + 1}')
tb_logger = SummaryWriter(path)

def create_full_domain_input_data(grid):
    """
    Creates an array that contains all combinations of time and space coordinates.
    """
    full_domain_input = np.array(np.meshgrid(grid.t, grid.x)).T.reshape(-1,2)
    full_domain_input = torch.Tensor(full_domain_input)
    return full_domain_input


def create_sensor_data(sim :HeatEqSimulation):
    # Input is an array with N_sensors*timesteps rows and 2 columns (time and sensor location)
    # Output is an array with N_sensors*timesteps rows and 2 column (sensor temperature data and source)


    sensor_input = None
    sensor_output = None

    

    for i in range(sim.number_of_sensors):
        #Concatenation of time and sensor location into one sensor_input array
        x_sensor = sim.sensor_location[i]*np.ones(sim.grid.temporal_gridpoints)
        x_sensor = x_sensor.reshape(sim.grid.temporal_gridpoints,1)

        arr = np.hstack((sim.grid.t.reshape(sim.grid.temporal_gridpoints,1), x_sensor))

        if sensor_input is None:
            sensor_input = arr
        else:
            sensor_input = np.vstack((sensor_input, arr))

        source = sim.source.source_field[np.argwhere(sim.grid.x == sim.sensor_location[i]),:].reshape(sim.grid.temporal_gridpoints,1)

        #Concatenation of sensor data into one sensor_output array
        sensor_data = sim.sensor_data[i,:].reshape(sim.grid.temporal_gridpoints,1)
        sensor_data = np.hstack((sensor_data, source))
        if sensor_output is None:
            sensor_output = sensor_data
        else:
            sensor_output = np.vstack((sensor_output, sensor_data))

    sensor_input = torch.Tensor(sensor_input)
    sensor_output = torch.Tensor(sensor_output)
    return sensor_input, sensor_output

def loss_function(T_pred, output, T_t, T_xx, source_strength_pred, hparam,T_0,T_left,T_right,T_NN_0,T_NN_left,T_NN_right,sim,epoch,batch_idx):
    
    # Physical Loss
    physical_loss = hparam["weight_physical"]*torch.mean((T_t - sim.conductivity*T_xx - source_strength_pred)**2)

    # Data Loss
    data_loss = hparam["weight_data"]*torch.mean((T_pred - output[:,0])**2)

    initial_loss = hparam["weight_initial"]*torch.mean((T_0 - T_NN_0)**2)

    boundary_loss = hparam["weight_boundary"]*torch.mean(((T_left - T_NN_left) + (T_right-T_NN_right))**2)

    current_learning_step = epoch * hparam["data_loader_length"] + batch_idx
    # Total Loss
    tb_logger.add_scalar('Loss/Physical', physical_loss, current_learning_step)
    tb_logger.add_scalar('Loss/Data', data_loss, current_learning_step)
    tb_logger.add_scalar('Loss/Initial', initial_loss, current_learning_step)
    tb_logger.add_scalar('Loss/Boundary', boundary_loss, current_learning_step) 

    loss = data_loss+physical_loss + boundary_loss + initial_loss
    
    return loss


def train_model(sim :HeatEqSimulation):
    # setting the hyperparameters
    hparam = {"learning_rate": 0.0001,"epochs": 20, "weight_physical":1, "weight_data":3,
              "weight_boundary":1,"weight_initial":1,"batch_size_physical": 200}
    
    # Get the data and reshape it into the correct format. 
    sensor_input, sensor_output = create_sensor_data(sim)
    sensor_input.requires_grad = True
    sensor_input = sensor_input.to(device)
    sensor_output = sensor_output.to(device)

    full_domain_input = create_full_domain_input_data(sim.grid)
    full_domain_input.requires_grad = True
    full_domain_input = full_domain_input.to(device)

    # Create torch dataset from the full_domain_input data with no output
    test_dataset = TensorDataset(full_domain_input)
    test_loader = DataLoader(test_dataset, batch_size=hparam["batch_size_physical"], shuffle=True)
    hparam["data_loader_length"] = len(test_loader)
    # setting a seed for pytorch as well as one for numpy
    torch.manual_seed(2)
    np.random.seed(2)

   
    
    losses = np.zeros(hparam["epochs"])

    # Instantiate the model, optimizer, data loaders and loss function
    model = Net(hparam).to(device)
    optimizer = Adam(model.parameters(), lr=hparam["learning_rate"])
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)

    index_t0 = full_domain_input[:,0] == 0
    index_left = full_domain_input[:,1] == 0
    index_right = full_domain_input[:,1] == 1
    
    input_bc_left = full_domain_input[index_left]
    input_bc_right = full_domain_input[index_right]
    input_ic = full_domain_input[index_t0]

    T_0 = torch.Tensor(sim.temperature_field[:,0]).to(device)
    T_left = torch.Tensor(sim.temperature_field[-1,:]).to(device)
    T_right = torch.Tensor(sim.temperature_field[0,:]).to(device)

    # Training loop
    for epoch in range(hparam["epochs"]):
        training_loss = 0
        for batch_idx, full_domain_input in enumerate(test_loader):
            optimizer.zero_grad()

            # Forward pass
            T_pred = model(sensor_input).to(device)[:,0]

            # Boundary Conditions   
            T_NN_0 = model(input_ic).to(device)[:,0]
            T_NN_left = model(input_bc_left).to(device)[:,0]
            T_NN_right = model(input_bc_right).to(device)[:,0]

            #Take the data from tensor list
            full_domain_input =  full_domain_input[0] 
            full_domain_input = full_domain_input.to(device)

            full_domain_output = model(full_domain_input).to(device)

            T_pred_physical, source_strength_pred = full_domain_output[:,0], full_domain_output[:,1]
           
            # Automatic differentiation
            dT = grad(T_pred_physical.sum(), full_domain_input, retain_graph= True, create_graph=True)[0]
            T_x = dT[:,1]
            T_t = dT[:,0]
            T_xx = grad(T_x, full_domain_input,grad_outputs= torch.ones((hparam["batch_size_physical"])).to(device), retain_graph=True, create_graph=True)[0][:,1]
        
            # Loss calculation
            loss = loss_function(T_pred, sensor_output, T_t, T_xx, source_strength_pred, hparam, T_0, T_left, T_right, T_NN_0, T_NN_left, T_NN_right,sim,epoch,batch_idx)
           
            # Backward pass
            loss.backward(retain_graph=True)

            # Update parameters
            optimizer.step()
            scheduler.step()

             # Logging
            training_loss += loss.item()
            tb_logger.add_scalar('train_loss', loss.item(), epoch * len(test_loader) + batch_idx)
            
        losses[epoch] = training_loss/len(test_loader)

        if epoch % 1 == 0:
            print("Epoch: %d, Loss: %.7f" % (epoch, losses[epoch]))


    # Forward pass
    torch.save(model.state_dict(), "heatPinn.pt")
    
    return model, losses


def main():
    grid = Grid(spatial_gridpoints=100,temporal_gridpoints=1000)
    source = Source.from_single_location(grid, strength_value= 100)
    sim = HeatEqSimulation(source, grid)
    model, losses = train_model(sim)

    full_domain_test_data = create_full_domain_input_data(sim.grid).to(device)
    model.eval()
    #ssplit the array at ever 50th row
    output_pred = model(full_domain_test_data)
    T_pred = output_pred[:,0]
    T_pred = T_pred.detach().cpu().numpy()
    T_pred = T_pred.reshape(sim.grid.temporal_gridpoints,sim.grid.spatial_gridpoints).T


    source_prediction = output_pred[:,1].detach().cpu().numpy().reshape(sim.grid.temporal_gridpoints,sim.grid.spatial_gridpoints).T
    plt.figure()   
    plt.imshow(source_prediction)
    plt.show()

    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.figure()

    plt.title("Predicted Temperature Field")
    plt.imshow(T_pred, cmap='hot', interpolation='nearest',aspect='auto')
    plt.colorbar()


    plt.figure()
    plt.title("True Temperature Field")
    plt.imshow(sim.temperature_field, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.show()
  

if __name__ == "__main__":

    main()