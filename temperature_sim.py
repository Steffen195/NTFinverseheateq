import numpy as np
import matplotlib.pyplot as plt
from utils import plot_heat_map, scale_temperature_field


class Grid():
    """
    Grid class describes objects that contain the coordinate grid for the simulation. 1 spatial dimension + 1 temporal dimension.
    """

    def __init__(self,spatial_gridpoints=50,x_left = -1, x_right = 1,temporal_gridpoints = 500, t_initial = 0, t_final = 1) -> None:
        self.spatial_gridpoints = spatial_gridpoints
        self.x_left = x_left
        self.x_right = x_right
        self.temporal_gridpoints = temporal_gridpoints
        self.t_initial = t_initial
        self.t_final = t_final
        
        self.dx = (self.x_right - self.x_left)/(self.spatial_gridpoints - 1)
        self.x = np.linspace(self.x_left, self.x_right, self.spatial_gridpoints)
        
        self.dt = (self.t_final - self.t_initial)/(self.temporal_gridpoints - 1)
        self.t = np.linspace(self.t_initial, self.t_final, self.temporal_gridpoints)
        
class Source():
    """
    Creates a source object that contains the source field with the same size as the given grid. 
    Can be constructed from a single location or from a function using different classmethods.
    """
    source_field : np.ndarray
    grid : Grid

    def __init__(self,source_field,grid,):
        self.grid = grid
        self.source_field = source_field 
        self.ratio = self.calculate_ratio()
    @classmethod
    def from_single_location(cls, grid: Grid,location= 1,strength_value = 1):
        source_field = np.zeros((grid.temporal_gridpoints,grid.spatial_gridpoints))
        source_strength =  np.ones(grid.temporal_gridpoints) * strength_value
        source_location_idx = int(np.floor(location / grid.dx)) # Wrong index due to new boundary condition
        source_field[:,source_location_idx] = source_strength
        
        return cls(source_field,grid)
    
    @classmethod
    def from_function(cls, grid: Grid, function= lambda t,x : max(0.0,np.exp(-1/2 * ((x**2)/0.1)) * 1/np.sqrt(2*np.pi*0.1**2)),threshold = None):
        t,x = np.meshgrid(grid.t,grid.x,indexing='ij')
        vfunction = np.vectorize(function)
        source_field = vfunction(t,x).reshape(grid.temporal_gridpoints,grid.spatial_gridpoints)  
        if np.max(source_field) > 1:
            source_field = source_field/np.max(source_field)

        if threshold is not None:
            source_field[source_field < threshold] = 0
        return cls(source_field,grid)
    
    def calculate_ratio(self):
        """
        This function calculates the fraction of non-zero values in the source field
        """
        ratio = np.sum(self.source_field != 0) / (self.grid.temporal_gridpoints * self.grid.spatial_gridpoints)
        return ratio


class HeatEqSimulation():
    """
    This class simulates the temperature distribution in a 1D rod using two Dirichlet boundary conditions
    and a source term. The source strength is a function of time. 
    """

    def __init__(self, source: Source, grid = Grid(), sensor_interval = 10, conductivity = 0.05, T_left = 0.5, T_right = 0.5, T_0 = 0.2):
        
        self.source = source
        self.grid = grid
        self.grid_check()
        self.sensor_interval = sensor_interval 
        self.sensor_location, self.number_of_sensors = self.set_sensor_location(sensor_interval)
        self.conductivity = conductivity
        self.stability_analysis()
        self.temperature_field = self.initialize_temperature_field(T_left, T_right,T_0)
        self.temperature_field = self.simulation()
        self.sensor_data = self.return_sensor_data()


    def grid_check(self):
        if self.source.grid != self.grid:
            raise ValueError("Source grid and domain grid do not match")
        
    def set_sensor_location(self,sensor_interval):
        sensor_location = self.grid.x[sensor_interval::sensor_interval]
        number_of_sensors = len(sensor_location)
        return sensor_location, number_of_sensors  
        
    def stability_analysis(self):
        """
        This function performs a stability analysis on the heat equation
        """
        stability_constant = self.conductivity * self.grid.dt / self.grid.dx**2

        if stability_constant > 0.5:
            raise ValueError("Stability condition not met. Stability constant is {}".format(
            stability_constant))
        else:
            print("Stability condition met. Stability constant is {}".format(stability_constant))



    def initialize_temperature_field(self,T_left, T_right, T_0):
        """
        This function initializes the temperature field
        """
        temperature_field = np.zeros((self.grid.temporal_gridpoints,self.grid.spatial_gridpoints))
        temperature_field[0, :] = T_0
        temperature_field[:, 0] = T_left
        temperature_field[:,-1] = T_right

        return temperature_field
    

    def simulation(self):
        """
        This function simulates the temperature distribution in a 1D rod using two Dirichlet boundary conditions
        and a source term. The source strength is a function of time. 
        """
        A = self.construct_A_matrix()
        identity = np.eye(self.grid.spatial_gridpoints)
        T = self.temperature_field
        # Solve the heat equation for each time step
        for time_index in range(1, self.grid.temporal_gridpoints):
            b_vector = self.construct_b_vector(time_index)
            T[time_index,:] = (identity + A) @ T[time_index-1,:] + b_vector
        T = scale_temperature_field(T)
        return T



    def construct_A_matrix(self):
        """
        This function constructs the matrix to discretize the heat equation
        """
        a_coeff = -2
        b_coeff = 1
        c_coeff = 1

        A = np.diag(np.ones(self.grid.spatial_gridpoints)*a_coeff) + np.diag(np.ones(self.grid.spatial_gridpoints-1)
                                                        * b_coeff, k=1) + np.diag(np.ones(self.grid.spatial_gridpoints-1)*c_coeff, k=-1)
        A = A * self.conductivity * self.grid.dt / self.grid.dx**2

        # Set the Dirichlet boundary conditions, where dT/dt = 0
        A[0, :] = 0
        A[-1, :] = 0

        return A


    def construct_b_vector(self, time_index):
        """
        This function constructs the b_vector on the right hand side of the heat equation. 
        Using the time_index, the source_strength at that time is extracted and placed in the b_vector
        at the sensor_location.
        """
        b_vector = self.source.source_field[time_index,:]
        return b_vector
    
    def return_sensor_data(self):
        """
        This function returns the sensor data at the sensor location
        """
        sensor_data = self.temperature_field[:,self.sensor_interval::self.sensor_interval]
        return sensor_data
        
def main():
    # Define the grid
    grid = Grid()
    # Define the source
    # lambda function with zero mean gaussian 
    
    #source = Source.from_single_location(grid)
    #function = lambda t,x : max([0.0,-3*x**2+0.3])
    #function = lambda t,x : np.exp(-1/2 * (((x-np.sin(50/(2*np.pi)*t))**2)/0.1)) * 1/np.sqrt(2*np.pi*0.1**2)
    source = Source.from_function(grid)#,function,threshold = 0.00)
    print(source.ratio)
    # Define the simulation
    simulation = HeatEqSimulation(source, grid)
    plot_heat_map(simulation.grid.x,simulation.grid.t,simulation.temperature_field)
    print(simulation.sensor_location)
    plt.imshow(source.source_field)
    plt.show()


if __name__ == "__main__":
    main()
