#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:12:06 2025

@author: alexolza
"""

import sys
sys.path.append('..')
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
###########################
from components.generators import VAE
from config_files.traditional_decnef_n_instances import traditional_decnef_n_instances_parser
#%%
"""
Configuration variables
"""
global_random_seed = 42
config = traditional_decnef_n_instances_parser()

device = 'cuda'
outpath = f'../EXPERIMENTS/{config.EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{config.EXP_NAME}/weights/'

z_dim = 2
generator_epochs = 20
generator_batch_size=64

generator_name = f'{config.generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)

#%% 
for p in [outpath, modelpath]:
    if not os.path.exists(p):
        os.makedirs(p)
#%%
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the MNIST training data
trainset = datasets.FashionMNIST('../data', download=True, train=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the MNIST test data
testset = datasets.FashionMNIST('../data', download=True, train=False, transform=transform)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)

if hasattr(trainset, 'class_to_idx'):
    class_name_dict = trainset.class_to_idx
else:
    class_name_dict = {v: v for v in trainset.classes}
    
class_names, class_numbers = np.array([[k,v] for k,v in class_name_dict.items()]).T
class_numbers = class_numbers.astype(int)
img_size = trainset[0][0].shape[-1]
#%%
if not os.path.exists(generator_fname+'.pt'):
    vae = VAE(z_dim=z_dim).to(device)
    vae.fit(train_loader, generator_epochs, generator_batch_size)
    vae.compute_prototypes(train_loader)
    vae_history = vae.history_to_df()
    print(f'{generator_name} TRAINING FINISHED WITH z_dim=',z_dim)
    #%%
    vae.save(generator_fname+'.pt')
    print(f'Saved {generator_fname}, exiting')
else:
    print(f'Found {generator_fname}, exiting')


