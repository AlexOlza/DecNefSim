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
#%%
"""
Configuration variables
"""
global_random_seed = 42
read_args = int(eval(sys.argv[1])) # Whether to use manual parameters, or read args
if read_args==1:
    EXP_NAME = sys.argv[2] # Output directory
    trajectory_random_seed = int(eval(sys.argv[3])) # Each trajectory will be determined by this variable
    target_class_idx = int(eval(sys.argv[4])) # The induction target for DecNef 
    non_target_class_idx = int(eval(sys.argv[5])) # The alternative class to train the binary discriminator
    lambda_inv = int(eval(sys.argv[6])) # The inverse of lambda, which controls the subject's ability to focus
    gamma_inv = int(eval(sys.argv[7])) # The inverse of gamma, which controls the subject's ability to react to the feedback
    decnef_iters = int(eval(sys.argv[8])) # DecNef loop iterations
    ignore_discriminator = int(eval(sys.argv[9])) # Whether to produce random feedback (for validation)
    update_rule_idx = int(eval(sys.argv[11])) # Select update rule from list
    production = int(eval(sys.argv[11])) # Whether this execution is a trial or a definitive one
    generator_name = 'VAE'
    discriminator_type = 'CNN'
    ext = 'png' if production==0 else 'pdf'
else:
    EXP_NAME = 'trash'
    trajectory_random_seed = 1
    target_class_idx = 0 
    non_target_class_idx = 1
    lambda_inv = 40
    gamma_inv = 40 
    decnef_iters = 500
    ignore_discriminator = 0
    update_rule_idx = 0
    production = 0 
    generator_name = 'VAE'
    discriminator_type = 'CNN'
    ext = 'png' if production==0 else 'pdf'


outpath = f'../EXPERIMENTS/{EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{EXP_NAME}/weights/'

z_dim = 2
lambda_ = 1/lambda_inv # A common value could be 0.025 which is 1/40
generator_epochs = 20
device='cuda'
generator_batch_size=64
tgt_non_tgt = [target_class_idx, non_target_class_idx]


generator_name = f'{generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
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


