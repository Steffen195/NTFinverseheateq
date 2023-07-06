from heateqsim import return_data
import numpy as np
import torch

def reshape_data(t,sensor_data, sensor_location):
    # Input is an array with N_sensors*timesteps rows and 2 columns (time and sensor location)
    # Output is an array with N_sensors*timesteps rows and 1 column (sensor temperature data)

    input = None
    output = None
    number_of_sensors = len(sensor_location)
    timesteps = len(t)
    t = t.reshape(timesteps,1)
    for i in range(number_of_sensors):
        #Concatenation of time and sensor location into one input array
        x_sensor = sensor_location[i]*np.ones(timesteps)
        x_sensor = x_sensor.reshape(timesteps,1)
        arr = np.hstack((t, x_sensor))
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

    return input, output

def train_model():
    # Get the data and reshape it into the correct format. 
    t, sensor_data, sensor_location, source_strength = return_data()
    input, output = reshape_data(t,sensor_data, sensor_location)


     # setting a seed for pytorch as well as one for numpy
    torch.manual_seed(2)
    np.random.seed(2)

    # setting the hyperparameters
    hparam = {"learning_rate": 0.0001}



def main():
    train_model()


if __name__ == "__main__":
    main()