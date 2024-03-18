# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 07:58:25 2023

@author: CHENE
"""
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from datetime import date
import numpy as np
from copy import deepcopy

import sys
import os
sys.path.append("./lib")

import Function as FX
from VAE_model import VariationalAutoEncoder

# Load dataset
ep_begin = 0
ep_end   = 700
interval = 5
    
foldername     = "007_Q2_015_0768_T"
version_name   = "007_Q2_015_0768_T_int5"     

path0, F0, extractor = FX.initialize_any(foldername + "/")
time, features, dList0 = FX.load_any(ep_begin, 
                                     ep_end, 
                                     F0, 
                                     interval=interval, 
                                     title='histories_short_print_')
ep_raw   = list()

# Normalization of data between 0 and 1 for VAE training
raw_data = np.vstack(dList0)[:,[3,8,4]]
TF_min   = np.min(raw_data[:,0])
TF_max   = np.max(raw_data[:,0])
UPT_min  = np.min(raw_data[:,1])
UPT_max  = np.max(raw_data[:,1])
FCL_min  = np.min(raw_data[:,2])
FCL_max  = np.max(raw_data[:,2])

for i in range(len(dList0)):
    temp = dList0[i][:,[3,8,4]]
    temp[:,0] = (temp[:,0]-TF_min)/(TF_max-TF_min)
    temp[:,1] = (temp[:,1]-UPT_min)/(UPT_max-UPT_min)
    temp[:,2] = (temp[:,2]-FCL_min)/(FCL_max-FCL_min)
    ep_raw.append(temp)
    
# Configuration of hyperparameters
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM    = 6006
H_DIM        = 256
Z_DIM        = 6
NUM_EPOCH    = 10000
BATCH_SIZE   = 128
weight_decay = 0
LR_RATE      = 1e-4   

# Dataset Loading
model        = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer    = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=LR_RATE)
loss_fn      = nn.BCELoss(reduction="sum")
curr_loss    = 100000

# Start Training
loop = tqdm(range(NUM_EPOCH))

for epoch in loop:
    for i in range(len(ep_raw)):
        # Forward Pass
        x                    = torch.from_numpy(ep_raw[i]).to(DEVICE).to(torch.float32)
        x                    = x.reshape(1,-1)
        #x                    = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconst, mu, sigma = model(x)
        
        # Compute loss
        reconst_loss = loss_fn(x_reconst, x)
        KL_div       = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) 
        w1   = 1     #reconstruction loss scaler  
        w2   = 0.5   #KL divergence loss scaler
        loss = w1*reconst_loss + w2*KL_div
        
        # Save best model and loss characteristics
        if loss.item() < curr_loss:
            best_model   = deepcopy(model.state_dict()) 
            best_loss    = loss.item()
            best_reconst = reconst_loss.item()
            best_KL_div  = KL_div.item()
            curr_loss = loss.item()
            
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    loop.set_postfix(loss=loss.item())

# Setup folder for model settings
root      = './VAEModelSetting/'
modelname = 'VAE_' + version_name
directory = root + modelname + "/"
os.makedirs(directory, exist_ok=True)

# Save model and parameter configurations
torch.save(best_model.state_dict(), directory+modelname+".pt")

save_shape = np.array([INPUT_DIM, H_DIM, Z_DIM]).reshape(1,-1)
np.savetxt(directory + modelname + "_Shape.txt", save_shape, delimiter=',')

# Save model development information
save_txt  = list()
save_txt.append("******************MODEL CONFIGURATION******************"+ "\n")
save_txt.append("Date of Creation: " + str(date.today())+ "\n")
save_txt.append("Model name: " + modelname + "\n"+ "\n")

save_txt.append("Input dimension: " + str(INPUT_DIM)+ "\n")
save_txt.append("Hidden dimension: " + str(H_DIM)+ "\n")
save_txt.append("Latent space dimension: " + str(Z_DIM) + "\n"+ "\n")
save_txt.append("******************TRAINING CONFIGURATION******************"+ "\n")
save_txt.append("Number of Epochs: " + str(NUM_EPOCH)+ "\n")
save_txt.append("Learning rate: " + str(LR_RATE)+ "\n")
save_txt.append("Loss function: Binary Cross Entropy"+ "\n")
save_txt.append("Training set size: " + str(int((ep_end-ep_begin)/interval))+ "\n")
save_txt.append("Training set origin: " + foldername+ "\n")
save_txt.append("KL divergence function lose scaler: " + str(w2)+ "\n")
save_txt.append("Reconstruction loss scaler: " + str(w1)+ "\n")
save_txt.append("Reconstruction error: " + str(best_reconst) + "\n")
save_txt.append("KL divergence error: " + str(best_KL_div) + "\n")
save_txt.append("Total reconstruction error: " + str(best_loss) + "\n")

# Write data to text file
with open(directory + modelname + '_Specifications.txt', 'w') as f:
    f.writelines(save_txt)

# Save engineering scaling constants
data_2    = np.array([TF_max, 
                      TF_min, 
                      UPT_max, 
                      UPT_min, 
                      FCL_max, 
                      FCL_min]).reshape(1,-1)

np.savetxt(directory + modelname + "_ESV.txt", 
           data_2, 
           delimiter=',', 
           header="Total flow max, Total flow min, Upper plenum max, Upper plenum min, Fuel centerline max, Fuel centerline min")

    



