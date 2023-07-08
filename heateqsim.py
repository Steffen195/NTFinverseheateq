import numpy as np
import matplotlib.pyplot as plt


def stability_analysis(conductivity, dx, dt):
    """
    This function performs a stability analysis on the heat equation
    """
    stability_constant = conductivity * dt / dx**2
    if stability_constant > 0.5:
        raise ValueError("Stability condition not met. Stability constant is {}".format(
            stability_constant))
    else:
        print("Stability condition met. Stability constant is {}".format(
            stability_constant))


def construct_matrix(conductivity, dx, dt, grid_size):
    """
    This function constructs the matrix to discretize the heat equation
    """
    a_coeff = -2
    b_coeff = 1
    c_coeff = 1

    A = np.diag(np.ones(grid_size)*a_coeff) + np.diag(np.ones(grid_size-1)
                                                      * b_coeff, k=1) + np.diag(np.ones(grid_size-1)*c_coeff, k=-1)
    A = A * conductivity * dt / dx**2

    # Set the Dirichlet boundary conditions, where dT/dt = 0
    A[0, :] = 0
    A[-1, :] = 0

    return A


def construct_b_vector(source_strength, time_index, source_location, grid_size):
    """
    This function constructs the b_vector on the right hand side of the heat equation. 
    Using the time_index, the source_strength at that time is extracted and placed in the b_vector
    at the sensor_location.
    """
    b_vector = np.zeros(grid_size)
    b_vector[source_location] = source_strength[time_index]
    return b_vector


def extract_sensor_data(T, sensor_location_index):
    """
    This function extracts the sensor data from the Temperature array
    at the specified locations of the sensors.
    """
    sensor_data = T[sensor_location_index, :]
    return sensor_data


def temperature_simulation():
    """
    This function simulates the temperature distribution in a 1D rod using two Dirichlet boundary conditions
    and a source term. The source strength is a function of time. 
    """
    conductivity = 0.05

    # Set up the grid
    x_left_boundary = 0
    x_right_boundary = 1
    grid_size = 100
    dx = (x_right_boundary - x_left_boundary) / grid_size
    x = np.linspace(x_left_boundary, x_right_boundary, grid_size)

    # Set up the time grid
    t0 = 0
    t_final = 1
    time_steps = 1000
    dt = (t_final - t0) / time_steps
    t = np.linspace(t0, t_final, time_steps).T


    # Source term. Each element i is the source at time t_i
    source_strength = np.ones(time_steps) * 1
    source_location = int(np.floor(grid_size / 2)-1)
    

    # Perform stability analysis
    stability_analysis(conductivity, dx, dt)

    # Set up the Temperature array
    T = np.ones((grid_size, time_steps))

    # Initial Conditions
    T[:, 0] = 2

    # Boundary Conditions
    T_left = 5
    T_right = 5

    T[0, :] = T_left
    T[-1, :] = T_right

    # Set up the matrix to discretize the heat equation with source term
    A = construct_matrix(conductivity, dx, dt, grid_size)

    identity = np.eye(grid_size)
    # Solve the heat equation for each time step
    for time_index in range(1, time_steps):
        b_vector = construct_b_vector(
            source_strength, time_index, source_location, grid_size)
        T[:, time_index] = (identity + A) @ T[:, time_index-1] + b_vector

    return x, t, T, source_location, source_strength


def return_data():
    x, t, T,source_location, source_strength = temperature_simulation()
    # Extract Sensor Data

    
    #Select every fifth entry in the x array
    sensor_location = x[::10]
    sensor_location_index = [int(np.floor(i * len(x)))
                             for i in sensor_location]

    sensor_data = extract_sensor_data(T, sensor_location_index)

    return x,t, T, sensor_data, sensor_location, source_location,source_strength

def plot_heat_map(x,t,T):
    # Plot the results in heat map
    plt.figure()
    plt.imshow(T, cmap='hot', interpolation='nearest',
               extent=[t[0],t[-1],x[0],x[-1]], aspect='auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.colorbar()
    plt.title('Temperature Distribution in 1D Rod')
    plt.show()


def main():
    x,t,T,_,_ = temperature_simulation()
    # Plot the results in heat map
    plot_heat_map(x,t,T)




if __name__ == "__main__":
    main()
