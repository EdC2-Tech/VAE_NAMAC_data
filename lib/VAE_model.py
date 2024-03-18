# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:13:23 2023

@author: CHENE
"""

import torch
import torch.nn as nn

# define a simple linear VAE
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        
        # Encoder
        self.img_2hid   = nn.Linear(input_dim, hidden_dim)
        self.hid_2mu    = nn.Linear(hidden_dim, z_dim)
        self.hid_2sigma = nn.Linear(hidden_dim, z_dim)
        
        # Decoder
        self.z_2hid     = nn.Linear(z_dim, hidden_dim)
        self.hid_2img   = nn.Linear(hidden_dim, input_dim)

        self.relu       = nn.ReLU()
        
    def encode(self, sample):
        h = self.relu(self.img_2hid(sample))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma 
        
    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))
    
    def forward(self, sample):
        mu, sigma = self.encode(sample)
        epsilon   = torch.randn_like(sigma)
        z_reparam = mu + sigma*epsilon
        x_reconst = self.decode(z_reparam)
        return x_reconst, mu, sigma