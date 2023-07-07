import os
from heateqsim import return_data
import numpy as np
import torch
from model.network import Net
from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using the following device: ', device)

def create_test_data(x,t,source_strength,source_location):

    input_test = np.array(np.meshgrid(t,x)).T.reshape(-1,2)
    test_source = np.zeros((np.shape(input_test)[0],1))
    for i in range(len(input_test)-1):
        if input_test[i,1] == source_location:
            test_source[i] = source_strength(np.where(t==input_test[i,0]))

    input_test = np.hstack((input_test, test_source))
    input_test = torch.Tensor(input_test)

    return input_test




def reshape_data(t,sensor_data, sensor_location,source_strength):
    # Input is an array with N_sensors*timesteps rows and 2 columns (time and sensor location)
    # Output is an array with N_sensors*timesteps rows and 1 column (sensor temperature data)

    input = None
    output = None
    source_strength = source_strength.reshape(len(source_strength),1)
    number_of_sensors = len(sensor_location)
    
    timesteps = len(t)
    t = t.reshape(timesteps,1)
    for i in range(number_of_sensors):
        #Concatenation of time and sensor location into one input array
        x_sensor = sensor_location[i]*np.ones(timesteps)
        x_sensor = x_sensor.reshape(timesteps,1)

        if i == 1:
            source = source_strength
        else:
            source = np.zeros_like(t)

        arr = np.hstack((t, x_sensor,source))

        if input is None:
            input = arr
        else:
            input = np.vstack((input, arr))

        #Concatenation of sensor data into one output array
        data = sensor_data[i,:].reshape(timesteps,1)
        if output is None:
            output = data
        else:
            output = np.vstack((output, data))


    input = torch.Tensor(input)
    output = torch.Tensor(output)
    return input, output

def reshape_val_data(x,t,temperature_field,source_strength,source_location):
    val_ratio = 0.2
    val_number = int(len(x)*len(t)*val_ratio)
    x_val = np.random.choice(x, val_number).reshape(val_number,1)
    t_val = np.random.choice(t, val_number).reshape(val_number,1)
    val_source = np.zeros_like(x_val)
    for i in range(len(x_val)-1):
        if x_val[i] == source_location:
            val_source[i] = source_strength(np.where(t==t_val[i]))
        else: 
            val_source[i] = 0.0

    val_input = np.hstack((t_val, x_val, val_source))

    val_output = temperature_field[np.where(x==x_val), np.where(t==t_val)]

    return val_input, val_output

def loss_function(T_pred, output, T_t, T_xx, source_strength, hparam,T_0,T_left,T_right,T_NN_0,T_NN_left,T_NN_right):
    # Physical Loss
    physical_loss = torch.mean((T_t - 0.05*T_xx - source_strength)**2)

    # Data Loss
    data_loss = torch.mean((T_pred - output)**2)

    initial_loss = torch.mean((T_0 - T_NN_0)**2)

    boundary_loss = torch.mean(((T_left - T_NN_left) + (T_right-T_NN_left))**2)

    # Total Loss
    #print("Physical loss: ", physical_loss) 
    #print("Data loss: ", data_loss)
    loss = hparam["weight_data"]*data_loss+hparam["weight_physical"]*physical_loss + hparam["weight_boundary"]*boundary_loss + hparam["weight_initial"]*initial_loss
    return loss

def train_model():
    # Get the data and reshape it into the correct format. 
    x, t, temperature_field, sensor_data, sensor_location, source_location, source_strength = return_data()
    input, output = reshape_data(t,sensor_data, sensor_location,source_strength)
    input.requires_grad = True
    input = input.to(device)
    output = output.to(device)
    input_test = create_test_data(x,t,source_strength,source_location)
    input_test.requires_grad = True
    input_test = input_test.to(device)


    #val_input, val_output = reshape_val_data(x,t,temperature_field,source_strength,source_location)
    # setting a seed for pytorch as well as one for numpy
    torch.manual_seed(2)
    np.random.seed(2)

    # setting up the tensorboard logger
    path = os.path.join('logs', 'Case_0')
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path = os.path.join(path, f'run_{num_of_runs + 1}')
    tb_logger = SummaryWriter(path)
    
    # setting the hyperparameters
    hparam = {"learning_rate": 0.001,"epochs": 700, "weight_physical":0.01, "weight_data":2,"weight_boundary":1,"weight_initial":1}

    losses = np.zeros(hparam["epochs"])
    # Get validation data by drawing random samples from the full temperature field

    # Instantiate the model, optimizer, data loaders and loss function
    model = Net(hparam).to(device)
    optimizer = Adam(model.parameters(), lr=hparam["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1000, gamma=1)


    index_t0 = input_test[:,0] == 0
    index_left = input_test[:,1] == 0
    index_right = input_test[:,1] == 1

    T_0 = torch.Tensor(temperature_field[:,0]).to(device)
    T_left = torch.Tensor(temperature_field[-1,1:]).to(device)
    T_right = torch.Tensor(temperature_field[0,1:]).to(device)

    # Training loop
    for epoch in range(hparam["epochs"]):
        training_loss = 0
        
        # Forward pass
        T_pred = model(input).to(device)
        T_pred_physical = model(input_test).to(device)

        # Boundary Conditions   
        T_NN_0 = T_pred_physical[index_t0]
        T_NN_left = T_pred_physical[index_left][1:]
        T_NN_right = T_pred_physical[index_right][1:]
 
        # Automatic differentiation
        dT = grad(T_pred_physical.sum(), input_test, create_graph=True)[0]
        T_x = dT[:,1]
        T_t = dT[:,0]
        T_xx = grad(T_x, input_test,grad_outputs= torch.ones((100000)).to(device), create_graph=True)[0][:,1]
        
        optimizer.zero_grad()
        # Loss calculation

        loss = loss_function(T_pred, output, T_t, T_xx, input_test[:,2],hparam,T_0,T_left,T_right,T_NN_0,T_NN_left,T_NN_right)

        #trainig_loss += loss.item()

        # Logging
        tb_logger.add_scalar('train_loss', loss.item(), epoch)
        # Backward pass
        loss.backward()
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        losses[epoch - 1] = loss.detach().cpu().numpy()
        # Validation
        #T_val = model(val_input)
        if epoch % 100 == 0:
            print("Epoch: %d, Loss: %.7f" % (epoch, losses[epoch - 1]))


        # Forward pass
    torch.save(model.state_dict(), "heatPinn.pt")
    test_data = create_test_data(x,t,source_strength,source_location).to(device)
    return model, losses,test_data, temperature_field


def main():
    model, losses,test_data, T= train_model()
    model.eval()
    #ssplit the array at ever 50th row
    T_pred = model(test_data)
    T_pred = T_pred.detach().cpu().numpy()
    T_pred = T_pred.reshape(1000,100).T
        


    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.figure()

    plt.title("Predicted Temperature Field")
    plt.imshow(T_pred, cmap='hot', interpolation='nearest',aspect='auto')


    plt.figure()
    plt.title("True Temperature Field")
    plt.imshow(T, cmap='hot', interpolation='nearest', aspect='auto')
    plt.show()
  

if __name__ == "__main__":

    main()