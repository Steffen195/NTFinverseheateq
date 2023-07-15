import numpy as np
import matplotlib.pyplot as plt

def plot_heat_map(x,t,temperature_field):
    # Plot the results in heat map
    plt.figure()
    plt.imshow(temperature_field, cmap='hot', interpolation='nearest',
               extent=[t[0],t[-1],x[0],x[-1]], aspect='auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.colorbar()
    plt.title('Temperature Distribution in 1D Rod')
    plt.show()