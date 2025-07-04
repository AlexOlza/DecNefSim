#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 17:05:25 2025

@author: alexolza
"""
import os
import sys
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# from itertools import zip_longest
from torchvision import datasets
###########################################
from config_files.traditional_decnef_n_instances import traditional_decnef_n_instances_parser
from components.generators import VAE
from components.discriminators import CNNClassification
from visualization.plotting import evolution_with_CI, show_NANs
###########################################
# Computer Vision metrics that compare two images
from analysis.image_metrics import pixel_correlation, compute_ssim, metric_evolution

# CV metrics based on DL models
from analysis.image_metrics import two_way_identification
############################################
#%%
config = traditional_decnef_n_instances_parser()

n_trajs = 250
z_dim = 2
lambda_ = 1/config.lambda_inv # A common value could be 0.025 which is 1/40
generator_epochs = 20
device='cuda'
generator_batch_size=64

discriminator_epochs = 10
discriminator_batch_size = 16
tgt_non_tgt = [config.target_class_idx, config.non_target_class_idx]

outpath = f'../EXPERIMENTS/{config.EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{config.EXP_NAME}/weights/'


trainset = datasets.FashionMNIST('../data', download=True, train=True)
if hasattr(trainset, 'class_to_idx'):
    class_name_dict = trainset.class_to_idx
else:
    class_name_dict = {v: v for v in trainset.classes}
combo_names = [list(class_name_dict.keys())[i] for i in tgt_non_tgt]
clean_discr_str = re.sub('[^a-zA-Z0-9]','', f'{combo_names[0]} vs {combo_names[1]}')
figpath = f'../EXPERIMENTS/{config.EXP_NAME}/figures/{clean_discr_str}/nfb_eval/'

if not os.path.exists(figpath): os.makedirs(figpath)

discriminator_name = f'{config.discriminator_type}_{clean_discr_str}__BS{discriminator_batch_size}_E{discriminator_epochs}'
discriminator_fname = os.path.join(modelpath, discriminator_name+'.pt')
generator_name = f'{config.generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)
# %%
# discriminator_fname2 = '../EXPERIMENTS/test/weights/CNN_SandalvsPullover__BS16_E10.pt'
# dis2 = CNNClassification(torch.Size([1, 28, 28]), [5,2])
# dis2.load(discriminator_fname2)
# %%

   
discriminator = CNNClassification(torch.Size([1, 28, 28]), tgt_non_tgt)
discriminator.load(discriminator_fname)

generator = VAE(z_dim=z_dim)
generator.load(generator_fname+'.pt')

latent_prototype = generator.prototypes[config.target_class_idx][0] # [1] is the variance and [0] is the mu
prototype = generator.decoder(torch.Tensor(latent_prototype),
                                            generator.target_size).detach()

trajectory_dir = os.path.join(outpath,f'TRAJS_{generator_name}_{discriminator_name}',f'UR{config.update_rule_name}',f'IGDIS{config.ignore_discriminator}')

probability_matrix = np.full((config.decnef_iters+1, n_trajs), np.nan)
sigma_matrix = np.full((config.decnef_iters+1, n_trajs), np.nan)
pixcorr_matrix = np.full((config.decnef_iters+1, n_trajs), np.nan)
ssim_matrix = np.full((config.decnef_iters+1, n_trajs), np.nan)
two_way_identification_matrix = np.full((config.decnef_iters+1, n_trajs), np.nan)
trajectory_matrix = np.full((config.decnef_iters+1, n_trajs, 2), np.nan)

for i in range(n_trajs):
    trajectory_name = f'TRAJ{i}_{generator_name}_{discriminator_name}_UR{config.update_rule_name}_IGDIS{config.ignore_discriminator}'

    with np.load(os.path.join(trajectory_dir, f'{trajectory_name}.npz')) as trajectory:
        probability_matrix[:len(trajectory['probabilities']),i] = trajectory['probabilities']
        sigma_matrix[:len(trajectory['sigma']),i] = trajectory['sigma']
        
        trajectory_matrix[:len(trajectory['trajectory']),i] = trajectory['trajectory']
        
        gen_imgs = trajectory['generated_images']
        pixcorr_matrix[:len(gen_imgs),i] = metric_evolution(gen_imgs,
                                                           prototype,
                                                           pixel_correlation)
        ssim_matrix[:len(gen_imgs),i] = metric_evolution(gen_imgs,
                                                           prototype,
                                                           compute_ssim)
        # sns.heatmap(gen_imgs[500,0,:,:])

probability_df = pd.DataFrame(probability_matrix)
sigma_df = pd.DataFrame(sigma_matrix)
pixcorr_df = pd.DataFrame(pixcorr_matrix)
ssim_df = pd.DataFrame(ssim_matrix)
#%%
UR = 'MNDAVMem'
IGDIS=0
ext = 'png'
for df, ylabel, title, fname in zip([probability_df, sigma_df, pixcorr_df, ssim_df],
                             ['p(Y=y*|x)', 'sigma', 'PixCorr', 'SSIM'],
                             [f'Evolution of the probability (UR: {UR})',
                              f'Evolution of the variance (UR: {UR})',
                              f'Evolution of the Pixel Correlation (UR: {UR})',
                              f'Evolution of the Structural Similarity (UR: {UR})'],
                             [os.path.join(figpath, f'evol_probability_{UR}_IGDIS{IGDIS}.{ext}'),
                              os.path.join(figpath, f'evol_variance_{UR}_IGDIS{IGDIS}.{ext}'),
                              os.path.join(figpath, f'evol_pixcorr_{UR}_IGDIS{IGDIS}.{ext}'),
                              os.path.join(figpath, f'evol_ssim_{UR}_IGDIS{IGDIS}.{ext}'),]
                             ):
    evolution_with_CI(df, title, ylabel, fname=fname)
    show_NANs(df, ylabel)

#%%
import matplotlib.pyplot as plt
import numpy as np

# Example data: shape (time, samples, 2)
# 100 time steps, 250 samples, 2D position (x, y)
n_time, n_samples = 100, 250
# data = np.cumsum(np.random.randn(n_time, n_samples, 2), axis=0)  # random walk
data = trajectory_matrix
# Plot all trajectories in 2D space
plt.figure(figsize=(8, 8))
for i in range(n_trajs):
    plt.plot(data[:, i, 0], data[:, i, 1], alpha=0.1, color='blue')  # trajectory of sample i

# Overlay mean trajectory
mean_traj = data.mean(axis=1)
plt.plot(mean_traj[:, 0], mean_traj[:, 1], color='red', linewidth=2, label='Mean trajectory')

plt.xlabel('X position')
plt.ylabel('Y position')
plt.title(f'2D Trajectories Over Time (UR: {UR})')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
