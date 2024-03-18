# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:01:45 2023

@author: ChEdw
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("./lib")

from VAE_model import VariationalAutoEncoder

def infer(mu, sigma, time, sample_size):
    '''
    Generates n number of sampeles using latent space averages.

    Parameters
    ----------
    mu : tensor, array-like
        Latent space mean values.
    sigma : tensor, array-like
        Latent space standard deviation of mean values.
    time : ndarray
        Time scale for transient analysis
    sample_size : int
        Number of samples to generate.

    Returns
    -------
    out : list of ndarray
        List of datasamples

    '''    
    # Decode latent space
    out = list()
    
    for i in range(sample_size):
        # Reparameterize latent space
        epsilon = torch.randn_like(sigma)
        z = mu+sigma+epsilon
        
        # Generate data
        temp = model.decode(z)
        temp = temp.view(2002,3).cpu().detach().numpy()
        out.append(temp)
    
    return out
 
def rescale(raw_data, time, srcFile):
    '''
    Rescales data to engineering values. Do not modify

    Parameters
    ----------
    raw_data : ndarray
        Array of sensor and target values.
    time : ndarray
        Time scale axis of transient progression.

    Returns
    -------
    out_data : ndarray
        Transient data.

    '''
    # Engineering scalar constants, do not modify
    data     = np.genfromtxt(srcFile, delimiter=',', skip_header=True)
    TF_max   = data[0]
    TF_min   = data[1]
    UPT_max  = data[2]
    UPT_min  = data[3]
    FCL_max  = data[4]
    FCL_min  = data[5]
    
    # Rescale to engineering values
    for i in range(len(raw_data)):
        raw_data[i][:,0] = raw_data[i][:,0]*(TF_max-TF_min) + TF_min
        raw_data[i][:,1] = raw_data[i][:,1]*(UPT_max-UPT_min) + UPT_min
        raw_data[i][:,2] = raw_data[i][:,2]*(FCL_max-FCL_min) + FCL_min
        
        # Add time axis to datapoints
        raw_data[i] = np.hstack((time.reshape(-1,1), raw_data[i])) 
    
    return raw_data

if __name__ == "__main__":
    # Identify folder with dataset details
    foldername  = "VAE_007_Q2_015_0768_T_backup"
    modelname   = "VAE_007_Q2_015_0768_T_backup"
    directory   = './VAEModelSetting/'
    
    # Load variational autoencoder & structure
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelshape  = np.loadtxt("./VAEModelSetting/" + modelname + "/" + modelname + "_Shape.txt", delimiter=',', dtype=int)
    model       = VariationalAutoEncoder(modelshape[0], modelshape[1], modelshape[2]).to(DEVICE)
    state_dict  = torch.load(directory+modelname+"/"+modelname+".pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    # Determine hidden state layer
    mu      = state_dict["hid_2mu.bias"].to('cpu').detach().numpy()
    sigma   = state_dict["hid_2sigma.bias"].to('cpu').detach().numpy()
    
    # Load time scale parameters
    time_ref    = "./VAEModelSetting/Time_axis.csv"
    time_param  = np.genfromtxt(time_ref, delimiter=',')
    
    # Load engineering scaler constants file
    scale_file  = "./VAEModelSetting/" + modelname + "/" + modelname + "_ESV.txt"
    
    # Convert to tensors
    mu_params   = torch.from_numpy(mu).to(DEVICE).to(torch.float32).flatten()
    sig_params  = torch.from_numpy(sigma).to(DEVICE).to(torch.float32).flatten()
    
    # Generate dataset
    export_data = infer(mu_params, sig_params, time_param, 10)
    export_data = rescale(export_data, time_param, scale_file)
    #np.savetxt("TransientData.csv", export_data, delimiter=',') 
    
    # Plot fuel centerline dataset
    plt.figure()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Fuel centerline temperature [C]")
        
    for case in export_data:
        plt.plot(case[:,0], case[:,3])
    
    # Plot flow rate dataset
    plt.figure()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Flow rate [kg/s]")
    
    for case in export_data:
        plt.plot(case[:,0], case[:,1])
        
    # Plot upper plenum  temperature dataset
    plt.figure()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Upper Plenum Temperature [C]")
    
    for case in export_data:
        plt.plot(case[:,0], case[:,2])
        plt.grid()
        
    # Plot input data
    plt.figure()
    plt.grid()
    plt.ylabel("Total Core Flow rate [kg/s] ")
    plt.xlabel("Upper Plenum Temperature [C]")
    
    for case in export_data:
        plt.scatter(case[:,2], case[:,1], color='k', s=2)
        plt.grid()    
        
        
