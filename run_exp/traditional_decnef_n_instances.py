#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 12:06:50 2025

@author: alexolza
"""

import sys
sys.path.append('..')
import torch
import os
from tqdm import tqdm
import regex as re
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
###########################
from components.generators import VAE
from protocols.decnef_loops import compute_single_trajectory
from components.discriminators import CNNClassification, BinaryDataLoader
from components.update_rules import update_z_moving_normal_drift_adaptive_variance, powsig, update_z_moving_normal_drift_adaptive_variance_memory
from config_files.traditional_decnef_n_instances import traditional_decnef_n_instances_parser
#%%
"""
Configuration variables
"""
global_random_seed = 42
p_scale_func = powsig
config = traditional_decnef_n_instances_parser()


figpath = f'../EXPERIMENTS/{config.EXP_NAME}/figures/'
outpath = f'../EXPERIMENTS/{config.EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{config.EXP_NAME}/weights/'
genfigpath = figpath+'generator_eval'
disfigpath = figpath+'discriminator_eval'
nfbfigpath = figpath+'nfb_eval'
update_rules = [update_z_moving_normal_drift_adaptive_variance, 
                update_z_moving_normal_drift_adaptive_variance_memory
                ]
update_rule_names = ['MNDAV', 'MNDAVMem']

z_dim = 2
lambda_ = 1/config.lambda_inv # A common value could be 0.025 which is 1/40
generator_epochs = 20
device='cuda'
generator_batch_size=64

discriminator_epochs = 10
discriminator_batch_size = 16
tgt_non_tgt = [config.target_class_idx, config.non_target_class_idx]

generator_name = f'{config.generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)

#%%
nfb_traj_figpath = os.path.join(nfbfigpath,'VAE2D_trajectories')
dis_pdist_figpath = os.path.join(disfigpath,'probabilities_by_classifier') 
for p in [genfigpath, dis_pdist_figpath,
          nfb_traj_figpath, 
          modelpath]:
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
else:
    vae = VAE(z_dim=z_dim).to(device)
    vae.load(generator_fname+'.pt')
    vae_history = vae.history_to_df()
    print(f'Loaded {generator_fname}')

update_rule_func, update_rule_name = update_rules[config.update_rule_idx], update_rule_names[config.update_rule_idx]

#%%
classes = trainset.targets.unique().numpy()
class_names = trainset.classes
combo_names = [list(class_names)[i] for i in tgt_non_tgt]
discr_str = f'{combo_names[0]} vs {combo_names[1]}'
clean_discr_str = re.sub('[^a-zA-Z0-9]','', discr_str)
discriminator_name = f'{config.discriminator_type}_{clean_discr_str}__BS{discriminator_batch_size}_E{discriminator_epochs}'
discriminator_fname = os.path.join(modelpath, discriminator_name+'.pt')
discriminator = CNNClassification(torch.Size([1, 28, 28]), tgt_non_tgt, device, name=discr_str) 
if not os.path.exists(discriminator_fname):
    print('DISCRIMINATOR TRAINING: ',discr_str)
    tl = BinaryDataLoader(trainset, tgt_non_tgt, batch_size=16)
    testl = BinaryDataLoader(testset, tgt_non_tgt, batch_size=16) 
    discriminator.evaluate(testl)
    discriminator.fit( epochs=discriminator_epochs, lr=1e-3, train_loader=tl, val_loader = testl)
    discriminator.save(discriminator_fname)
else:
    discriminator.load(discriminator_fname)
    discriminator.to(device)
    print(f'Loaded {discriminator_fname}')

trajectory_dir = os.path.join(outpath,f'TRAJS_{generator_name}_{discriminator_name}',f'UR{update_rule_name}',f'IGDIS{config.ignore_discriminator}')
if not os.path.exists(trajectory_dir): os.makedirs(trajectory_dir)
for trajectory_random_seed in range(config.trajectory_random_seed_init,
                                    config.trajectory_random_seed_init + config.n_trajectories + 1):
    #%%
    trajectory_name = f'TRAJ{trajectory_random_seed}_{generator_name}_{discriminator_name}_UR{update_rule_name}_IGDIS{config.ignore_discriminator}'
    trajectory_fname = os.path.join(trajectory_dir, f'{trajectory_name}.npz')
    if os.path.exists(trajectory_fname): print('Found'); continue
    generated_images,\
    trajectory,\
    probabilities,\
    all_probabilities,\
    sigma =  compute_single_trajectory(vae, discriminator,
                                       trajectory_random_seed,
                                       train_loader, config.target_class_idx,
                                       update_rule_func, p_scale_func,
                                       trajectory_name=trajectory_name, 
                                       n_iter = config.decnef_iters, lambda_ = lambda_,
                                       device=device, 
                                       ignore_discriminator=config.ignore_discriminator,
                                       start_from_origin=True,
                                       )
    #%%
    np.savez_compressed(trajectory_fname, 
                        generated_images = generated_images,
                        trajectory = trajectory,
                        probabilities = probabilities,
                        all_probabilities = all_probabilities,
                        sigma = sigma
                        )
    # print('Saved ',os.path.join(trajectory_dir, f'{trajectory_name}.npz'), ', exiting.')