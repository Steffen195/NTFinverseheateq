import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_heat_map(x,t,temperature_field):
    # Plot the results in heat map
    plt.figure()
    plt.imshow(temperature_field, cmap='hot', interpolation='nearest',
               extent=[x[0],x[-1],t[-1],t[0]], aspect='auto')
    plt.ylabel('Time (s)')
    plt.xlabel('Position (m)')
    plt.colorbar()
    plt.title('Temperature Distribution in 1D Rod')
    plt.show()


def scale_temperature_field(temperature_field):
    """
    This function scales the temperature field between 0 and 1
    """
    T_scaled = (temperature_field - np.min(temperature_field)) / (np.max(temperature_field) - np.min(temperature_field))
    return T_scaled


def create_full_domain_input_data(grid):
    """
    Creates an array that contains all combinations of time and space coordinates.
    """
    full_domain_input = np.array(np.meshgrid(grid.t, grid.x))
    full_domain_input = full_domain_input.T.reshape(-1,2)
    full_domain_input = torch.Tensor(full_domain_input)
    return full_domain_input

def create_collocation_points(n_collocation_points,grid):
    """
    Creates the collocation points for the evaluation of the physical loss.
    """
    time_to_space_ratio = grid.temporal_gridpoints / grid.spatial_gridpoints
    
    n_spatial_collocation_points = int(np.sqrt(n_collocation_points/time_to_space_ratio))
   
    n_temporal_collocation_points = int(np.floor(n_collocation_points/n_spatial_collocation_points))
    
   
    temporal_collocation_points = np.linspace(grid.t[0],grid.t[-1],n_temporal_collocation_points)
    
    spatial_collocation_points = np.linspace(grid.x[0],grid.x[-1],n_spatial_collocation_points)
    

    collocation_points = np.array(np.meshgrid(temporal_collocation_points, spatial_collocation_points))
    collocation_points = collocation_points.T.reshape(-1,2)
    collocation_points = torch.Tensor(collocation_points)
    return collocation_points


def create_sensor_data(sim):
    """
    Creates the sensor data sliced from the simulation for the evaluation of the data, initial and boundary loss.
    Input is an array with N_sensors*timesteps rows and 2 columns (time and sensor location)
    Output is an array with N_sensors*timesteps rows and 2 column (sensor temperature data and source)
    """
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

        source = sim.source.source_field[:,np.argwhere(sim.grid.x == sim.sensor_location[i])].reshape(sim.grid.temporal_gridpoints,1)

        #Concatenation of sensor data into one sensor_output array
        sensor_data = sim.sensor_data[:,i].reshape(sim.grid.temporal_gridpoints,1)
        sensor_data = np.hstack((sensor_data, source))
        if sensor_output is None:
            sensor_output = sensor_data
        else:
            sensor_output = np.vstack((sensor_output, sensor_data))

    sensor_input = torch.Tensor(sensor_input)
    sensor_output = torch.Tensor(sensor_output)
    return sensor_input, sensor_output