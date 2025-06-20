#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:22:00 2025

@author: alexolza
"""
import sys
sys.path.append('..')
import torch
import os
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
#%%
"""
Configuration variables
"""
global_random_seed = 42
p_scale_func = powsig
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
    production = int(eval(sys.argv[10])) # Whether this execution is a trial or a definitive one
    generator_name = 'VAE'
    discriminator_type = 'CNN'
    ext = 'png' if production==0 else 'pdf'
else:
    EXP_NAME = 'trash'
    trajectory_random_seed = 1
    target_class_idx = 3
    non_target_class_idx = 1
    lambda_inv = 40
    gamma_inv = 40 
    decnef_iters = 500
    ignore_discriminator = 1
    update_rule_idx = 0
    production = 0 
    generator_name = 'VAE'
    discriminator_type = 'CNN'
    ext = 'png' if production==0 else 'pdf'


figpath = f'../EXPERIMENTS/{EXP_NAME}/figures/'
outpath = f'../EXPERIMENTS/{EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{EXP_NAME}/weights/'
genfigpath = figpath+'generator_eval'
disfigpath = figpath+'discriminator_eval'
nfbfigpath = figpath+'nfb_eval'
update_rules = [update_z_moving_normal_drift_adaptive_variance, 
                update_z_moving_normal_drift_adaptive_variance_memory
                ]
update_rule_names = ['MNDAV', 'MNDAVMem']

z_dim = 2
lambda_ = 1/lambda_inv # A common value could be 0.025 which is 1/40
generator_epochs = 20
device='cuda'
generator_batch_size=64
update_rule_func, update_rule_name = update_rules[update_rule_idx], update_rule_names[update_rule_idx]

discriminator_epochs = 10
discriminator_batch_size = 16
tgt_non_tgt = [target_class_idx, non_target_class_idx]

generator_name = f'{generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
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
    # eval_figs = visual_eval_vae(vae, vae_history, z_dim, train_loader, class_names, class_numbers)
    # fignames = [f'{generator_name}_{name}.{ext}' for name in ['LOSS','REC','PROT','LATENT_VIS', 'LATENT_TRAV']]
    # for figure, figurename in zip(eval_figs, fignames):
    #     figure.savefig(os.path.join(genfigpath, figurename), format=ext)
else:
    vae = VAE(z_dim=z_dim).to(device)
    vae.load(generator_fname+'.pt')
    vae_history = vae.history_to_df()
    print(f'Loaded {generator_fname}')

#%%
classes = trainset.targets.unique().numpy()
class_names = trainset.classes
combo_names = [list(class_names)[i] for i in tgt_non_tgt]
discr_str = f'{combo_names[0]} vs {combo_names[1]}'
clean_discr_str = re.sub('[^a-zA-Z0-9]','', discr_str)
discriminator_name = f'{discriminator_type}_{clean_discr_str}__BS{discriminator_batch_size}_E{discriminator_epochs}'
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

#%%
trajectory_name = f'TRAJ{trajectory_random_seed}_{generator_name}_{discriminator_name}_UR{update_rule_name}_IGDIS{ignore_discriminator}'
generated_images,\
trajectory,\
probabilities,\
all_probabilities,\
sigma =  compute_single_trajectory(vae, discriminator,
                                   trajectory_random_seed,
                                   train_loader, target_class_idx,
                                   update_rule_func, p_scale_func,
                                   trajectory_name=trajectory_name, 
                                   n_iter = decnef_iters, lambda_ = lambda_,
                                   device=device, 
                                   ignore_discriminator=ignore_discriminator,
                                   start_from_origin=True,
                                   )
#%%
np.savez_compressed(outpath + f'{trajectory_name}.npz', 
                    generated_images = generated_images,
                    trajectory = trajectory,
                    probabilities = probabilities,
                    all_probabilities = all_probabilities,
                    sigma = sigma
                    )
print('Saved ',outpath + f'{trajectory_name}.npz', ', exiting.')
#%%
if not read_args:
    out = np.load(outpath +f'{trajectory_name}.npz')
    from visualization.plotting import show_image
    from matplotlib import pyplot as plt
    
    fig, axs = plt.subplots(1,10)
    for ax, i in zip(axs, [0,10,30,50,80, 100, 110, 120, 130, -1]):
        ax.imshow(out['generated_images'][i][0])
        
    pd.DataFrame(out['probabilities']).plot()
    pd.DataFrame(out['sigma']).plot()


    target_prototype = vae.prototypes[target_class_idx][0].reshape(1,-1)
    dist = pd.DataFrame([np.sum((trajectory[i].reshape(1,-1)-target_prototype)**2) for i in range(len(trajectory))])
