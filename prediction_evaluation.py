import numpy as np
import matplotlib.pyplot as plt
from temperature_sim import HeatEqSimulation, Grid, Source
import torch.nn as nn
import torch
from utils import create_full_domain_input_data

#change formatting of plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["CMU Sans"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["CMU"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

plt.rcParams.update({
  "text.latex.preamble": plt.rcParams["text.latex.preamble"].join(
    [
        r"\usepackage{amsmath}",
    ]
  )
})

def evaluate_model(sim:HeatEqSimulation, model : nn.Module,losses,device,plot = False):
    full_domain_test_data = create_full_domain_input_data(sim.grid).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output_pred = model(full_domain_test_data)
        T_pred = output_pred[:,0].detach().cpu().numpy().reshape(sim.grid.temporal_gridpoints,sim.grid.spatial_gridpoints)
        source_prediction = output_pred[:,1].detach().cpu().numpy().reshape(sim.grid.temporal_gridpoints,sim.grid.spatial_gridpoints)

    source_total_error = calculate_total_error(sim.source.source_field,source_prediction)
    temperature_total_error = calculate_total_error(sim.temperature_field,T_pred)
    source_mean_square_error = calculate_mean_square_error(sim.source.source_field,source_prediction)
    temperature_mean_square_error = calculate_mean_square_error(sim.temperature_field,T_pred)
    

    summary = {"source_total_error":source_total_error,
                "temperature_total_error":temperature_total_error,
                "source_mean_square_error":source_mean_square_error,
                "temperature_mean_square_error":temperature_mean_square_error}
    
    if losses is not None:
        final_loss = losses[-1]
        summary["final_loss"] = final_loss

        if plot == True:
            plot_loss(losses)

    if plot == True:
        plot_source_prediction(sim,source_prediction)
        plot_temperature_prediction(sim,T_pred)
        plot_source_slice(0.5,sim,source_prediction)
        
    return summary


def calculate_total_error(simulated_field,predicted_field):
    """
    This function calculates the total error between the simulated and predicted temperature fields
    """
    error = np.sum(np.abs(simulated_field-predicted_field))
    return error

def calculate_mean_square_error(simulated_field,predicted_field):
    """
    This function calculates the mean square error between the simulated and predicted temperature fields
    """
    error = 1/(simulated_field.shape[0]*simulated_field.shape[1]) * np.sum((simulated_field-predicted_field)**2)
    return error

def plot_source_prediction(sim:HeatEqSimulation,source_prediction):
    """
    Plots the source field and the predicted source field side by side with a colorbar
    """
    
    source_ground_truth = sim.source.source_field
    fig, ax = plt.subplots(1,2,figsize=(10,5),layout = "constrained")

    min_source = np.min((np.min(source_prediction),np.min(source_ground_truth)))
    max_source = np.max((np.max(source_prediction),np.max(source_ground_truth)))

    ax[0].set_title("Prediction Source")
    im1 = ax[0].imshow(source_prediction, interpolation='nearest',aspect='auto',vmin=min_source, vmax=max_source,extent=[sim.grid.x[0],sim.grid.x[-1],sim.grid.t[-1],sim.grid.t[0]])
    ax[0].set_xlabel("Position")
    ax[0].set_ylabel("Time")    
   
    ax[1].set_title("Ground truth Source")
    im2 = ax[1].imshow(source_ground_truth, interpolation='nearest', aspect='auto',vmin=min_source, vmax=max_source,extent=[sim.grid.x[0],sim.grid.x[-1],sim.grid.t[-1],sim.grid.t[0]])
    ax[1].set_xlabel("Position")
    ax[1].set_ylabel("Time")

    plt.colorbar(mappable=im1, ax = ax, label="Source Intensity")
    plt.show()
    return fig, ax

def plot_temperature_prediction(sim:HeatEqSimulation,temperature_prediction):
    """
    Plots the temperature field and the predicted temperature field side by side with a colorbar
    """
    temperature_ground_truth = sim.temperature_field

    fig, ax = plt.subplots(1,2,figsize=(10,5),layout = "constrained")
    min_temperature = np.min((np.min(temperature_prediction),np.min(temperature_ground_truth)))
    max_temperature = np.max((np.max(temperature_prediction),np.max(temperature_ground_truth)))

    ax[0].set_title("Prediction Temperature")
    im1 = ax[0].imshow(temperature_prediction,cmap = "hot", interpolation='nearest',aspect='auto',vmin=min_temperature, vmax=max_temperature,extent=[sim.grid.x[0],sim.grid.x[-1],sim.grid.t[-1],sim.grid.t[0]])
    ax[0].set_xlabel("Position")
    ax[0].set_ylabel("Time")    
   
    ax[1].set_title("Ground truth Temperature")
    im2 = ax[1].imshow(temperature_ground_truth,cmap = "hot", interpolation='nearest', aspect='auto',vmin=min_temperature, vmax=max_temperature,extent=[sim.grid.x[0],sim.grid.x[-1],sim.grid.t[-1],sim.grid.t[0]])
    ax[1].vlines(x = sim.sensor_location,ymin=0,ymax=1,colors='white',linestyles='dashed')
    ax[1].set_xlabel("Position")
    ax[1].set_ylabel("Time")

    plt.colorbar(mappable=im1, ax = ax, label="Temperature")
    plt.show()

    return fig, ax

def plot_source_slice(t, sim, predicted_source_field):
    """
    This function plots the predicted source field at a given time slice
    """
    time_index = (np.abs(sim.grid.t - t)).argmin()
    y_max = np.max((sim.source.source_field[time_index,:],predicted_source_field[time_index,:]))
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(sim.grid.x,predicted_source_field[time_index,:],label="Predicted Source")
    ax.plot(sim.grid.x,sim.source.source_field[time_index,:],label="Ground Truth Source")
    ax.vlines(x = sim.sensor_location,ymin=0,ymax=1.1*y_max,colors='black',linestyles='dashed',label="Sensor Location")
    ax.set_xlabel("Position")
    ax.set_ylabel("Source Intensity")
    ax.set_title("Source Intensity at t = {}".format(t))
    ax.set_xlim([sim.grid.x[0],sim.grid.x[-1]])
    ax.set_ylim(ymin=0,ymax=1.1*y_max)
    ax.legend()
    plt.show()


def plot_loss(losses):
    print("More detailed loss information is found in the tensorboard logs")
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()