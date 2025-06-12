#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 15:18:45 2025

@author: alexolza
"""
import sys
sys.path.append('..')
import torch
from tqdm import tqdm
import os
import regex as re
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
###########################
from components.generators import VAE, train_model, get_data_predictions, get_classes_mean
from visualization.plotting import visual_eval_vae
from protocols.decnef_loops import minimal_loop, show_trajectories
from components.discriminators import CNNClassification, BinaryDataLoader, plot_predicted_probability_distribution
from components.generators import dist_classmean_to_random
from components.update_rules import update_z_fixed_normal_drift, update_z_moving_normal_drift_adaptive_variance, powsig, identity_f_p
#%%
"""
Configuration variables
"""
EXP_NAME = 'trash'#sys.argv[1]
generator_name = 'VAE'
discriminator_type = 'CNN'
ext = 'png'

figpath = f'../EXPERIMENTS/{EXP_NAME}/figures/'
outpath = f'../EXPERIMENTS/{EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{EXP_NAME}/weights/'
genfigpath = figpath+'generator_eval'
disfigpath = figpath+'discriminator_eval'
nfbfigpath = figpath+'nfb_eval'

discriminator_fname__ = os.path.join(modelpath, discriminator_type)

z_dim = 2
lambda_ = 0.025
generator_epochs = 20
device='cuda'
generator_batch_size=64
generator_model = VAE(z_dim=z_dim).to(device)

discriminator_epochs = 10
discriminator_batch_size = 16

generator_name = f'{generator_name}_Z{z_dim}_BS{generator_batch_size}_E{generator_epochs}'
generator_fname = os.path.join(modelpath, generator_name)
#%%
nfb_traj_figpath = os.path.join(nfbfigpath,'VAE2D_trajectories')
dis_pdist_figpath = os.path.join(disfigpath,'probabilities_by_classifier') 
reconstruction_path = outpath + 'reconstructions'

for p in [genfigpath, dis_pdist_figpath,
          nfb_traj_figpath, reconstruction_path,
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
    vae, vae_history = train_model(generator_model, train_loader,generator_batch_size, epochs=generator_epochs, z_dim=z_dim)
    print(f'{generator_name} TRAINING FINISHED WITH z_dim=',z_dim)
    #%%
    torch.save(vae.state_dict(), generator_fname+'.pt')
    vae_history.to_csv(generator_fname+'_history.csv', index=False)
    eval_figs = visual_eval_vae(vae, vae_history, z_dim, train_loader, class_names, class_numbers)
    fignames = [f'{generator_name}_{name}.{ext}' for name in ['LOSS','REC','PROT','LATENT_VIS', 'LATENT_TRAV']]
    for figure, figurename in zip(eval_figs, fignames):
        figure.savefig(os.path.join(genfigpath, figurename), format=ext)
else:
    vae = generator_model
    state_dict = torch.load(generator_fname+'.pt')
    vae.load_state_dict(state_dict)
    vae.to(device)
    vae_history = pd.read_csv(generator_fname+'_history.csv')
    print(f'Loaded {generator_fname}')
#%%



#%%


def train_all_discriminators(train_loader, 
                             classes=None, class_names=None,
                             discr_epochs=discriminator_epochs,
                             batch_size= discriminator_batch_size,
                             discriminator_fname= discriminator_fname__):
    discr_dict = {}
    if classes is None: classes = trainset.targets.unique().numpy()
    if class_names is None: class_names = trainset.classes
    class_combinations = list(itertools.combinations(classes, 2))
    
    for combo in tqdm(class_combinations):
        combo_names = [list(class_names)[i] for i in combo]
        discr_str = f'{combo_names[0]} vs {combo_names[1]}'
        clean_discr_str = re.sub('[^a-zA-Z0-9]','', discr_str)
        discriminator_fname = f'{discriminator_fname__}_{clean_discr_str}__BS{discriminator_batch_size}_E{discr_epochs}.pt'
        discriminator = CNNClassification(torch.Size([1, 28, 28]), combo, device, name=discr_str) 
        if not os.path.exists(discriminator_fname):
            print('DISCRIMINATOR TRAINING: ',discr_str)
            tl = BinaryDataLoader(trainset, list(combo), batch_size=16)
            testl = BinaryDataLoader(testset, list(combo), batch_size=16) 
            discriminator.evaluate(testl)
            discriminator.fit( epochs=discr_epochs, lr=1e-3, train_loader=tl, val_loader = testl)
            discr_dict[discr_str] = discriminator
            torch.save(discriminator.state_dict(), discriminator_fname)
        else:
            state_dict = torch.load(discriminator_fname)
            discriminator.load_state_dict(state_dict)
            discriminator.to(device)
            discr_dict[discr_str] = discriminator
            print(f'Loaded {discriminator_fname}')
    return discr_dict

def combo_loops(combo, reverse, train_loader, generator, discriminator, lambda_, n_iter, device,  
                 class_names, classes_mean, update_rule_func, p_scale_func, z_current, random_state,
                 title='', title_color = 'black', n_str = '', 
                 **update_rule_kwargs):
    generated_images, trajectory, Dnorm, probabilities = {}, {}, {}, {}
    if not reverse: target_class, non_target_class = combo
    else:  non_target_class, target_class = combo
    
    # target_prototype = classes_mean[target_class]
    # non_target_prototype = classes_mean[non_target_class]
    target_class_name = class_names[target_class]
    # non_target_class_name = class_names[non_target_class]
    
    for d_str, ignore_discriminator in zip([f'p({target_class_name})', 'p~U(0,1)'], [0,2]):
        generated_images[d_str], trajectory[d_str], Dnorm[d_str], probabilities[d_str] = minimal_loop(train_loader, generator, discriminator, target_class, lambda_, n_iter, device,  
                         class_names, classes_mean, update_rule_func=update_rule_func, p_scale_func=p_scale_func, z_current=z_current, ignore_discriminator = ignore_discriminator, random_state=random_state, 
                         title=title, title_color = title_color, n_str = n_str, plot=False,
                         **update_rule_kwargs)
    return generated_images, trajectory, Dnorm, probabilities

def full_clasewise_DecNef(generator, discr_dict, train_loader, lambda_, n_iter, 
                          classes=None, class_names=None, reverse = False, z_current = None,
                          update_rule_func=update_z_fixed_normal_drift, p_scale_func=identity_f_p,
                          device='cuda',
                          random_state=0,
                          **p_scale_func_kwargs):
    if classes is None: classes = trainset.targets.unique().numpy()
    if class_names is None: class_names = trainset.classes
    latents_mean, latents_stdvar, labels = get_data_predictions(generator, train_loader)
    classes_mean = get_classes_mean(train_loader, labels, latents_mean, latents_stdvar) 
    
    generated_images, trajectory, Dnorm, probabilities = {}, {}, {}, {}
    
    class_combinations = list(itertools.combinations(classes, 2))
    for combo in tqdm(class_combinations): 
        combo_names = [list(class_names)[i] for i in combo]
        discr_str = f'{combo_names[0]} vs {combo_names[1]}'
        discriminator = discr_dict[discr_str]
        # Once the generator and the discriminator have been trained do the loop:
      
        generated_images[discr_str], trajectory[discr_str], Dnorm[discr_str], probabilities[discr_str] = combo_loops(combo, reverse, train_loader, generator, discriminator, lambda_, n_iter, device,  
                         class_names, classes_mean, update_rule_func, p_scale_func, z_current, random_state,
                         title='', title_color = 'black', n_str = '', 
                         **p_scale_func_kwargs)
            
    return generated_images, trajectory, Dnorm, probabilities



#%% TRAINING ALL DISCRIMINATORS
discr_dict = train_all_discriminators(train_loader, classes=None, class_names=None, discr_epochs=discriminator_epochs)

dist, dist0 = dist_classmean_to_random(trainset, class_numbers, class_names, device, niter=1000)   
probs, p0 = plot_predicted_probability_distribution(discr_dict, img_size, device, niter=1000)    

#%%
classes = trainset.targets.unique().numpy()
class_names = trainset.classes
class_combinations = list(itertools.combinations(classes, 2))
name_combinations = list(itertools.combinations(class_names, 2))
for c, cc in zip(name_combinations, class_combinations):
    reorder_classes = list(cc)+[i for i in range(10) if not i in cc]
    extra_classes = [class_numbers[i] for i in reorder_classes]# 
    extra_names = [class_names[i] for i in extra_classes]# 
    discr_str = f'{c[0]} vs {c[1]}'
    cols = f'{c[0]} vs {c[1]}'
    str_ = re.sub('/|\\s','-', discr_str)
    fname = os.path.join(dis_pdist_figpath, f"discr_ps_{str_}_violins.{ext}")
    
    e_ps={}
    for e, n in zip(extra_classes, extra_names):
        indices = (trainset.targets.clone().detach()[..., None] == e).any(-1).nonzero(as_tuple=True)[0]
        e_data = DataLoader(torch.utils.data.Subset(trainset, indices),batch_size=64)
        e_ps[n]= []
        for x, y in e_data:
            e_ps[n] = np.hstack([e_ps[n], torch.nn.Softmax()(discr_dict[discr_str](x.to(device))).T[0].cpu().detach().numpy()])

        e_ps[n] = np.array(e_ps[n]).flatten()
    e_ps['Gaussian noise']=  probs[discr_str].to_numpy().flatten()
    
    fig_violins, ax_violins = plt.subplots(figsize=(12,3))
    sns.violinplot(e_ps, cut=0, ax = ax_violins)
    ax_violins.axhline(y=0.5, color='grey',ls='--')
    fig_violins.tight_layout()
    fig_violins.savefig(fname)
#%%
n_iter = 10
lambda_ = 0.05
#%%
generated_images, trajectory, Dnorm, probabilities =  full_clasewise_DecNef(vae, discr_dict, train_loader, lambda_, n_iter, 
                                                                            classes=None, class_names=None, reverse = False, z_current = None,
                                                                            update_rule_func=update_z_fixed_normal_drift, p_scale_func=identity_f_p,
                                                                            device='cuda',
                                                                            random_state=0,
                                                                            )

#%%
generated_images_2d, trajectory_2d, Dnorm_2d, probabilities_2d =  full_clasewise_DecNef(vae, discr_dict, train_loader, lambda_, n_iter, 
                                                                            classes=None, class_names=None, reverse = False, z_current = None,
                                                                            update_rule_func=update_z_fixed_normal_drift, p_scale_func=identity_f_p,
                                                                            device='cuda',
                                                                            random_state=0,
                                                                            )
   
#%%
class_combinations___ = [(3, 6),(0, 2), (0, 3), (2, 8)]
name_combinations___  = [(class_names[c[0]], class_names[c[1]]) for c in class_combinations___]

for c, combo in zip(name_combinations___, class_combinations___):
    reorder_classes = list(combo)+[i for i in range(10) if not i in combo]
    extra_classes = [class_numbers[i] for i in reorder_classes]# 
    extra_names = [class_names[i] for i in reorder_classes]# 
    discr_str = f'{c[0]} vs {c[1]}'
    target_class = combo[0]
    print(discr_str)
    str_ = re.sub('/|\\s','-', discr_str)
    show_trajectories(vae, discr_dict[discr_str], train_loader, target_class, class_names,
                      update_z_moving_normal_drift_adaptive_variance,#update_z_fixed_normal_drift,
                      p_scale_func = powsig,#linsig,#f,#identity_f_p,
                      noise_sigma=torch.tensor(1.0),# starting_zs = None,
                      
                      filename = os.path.join(nfb_traj_figpath,f'{str_}_conv'),extension = 'pdf', zoom_radius=1.2,
                      n_iter = 600, n_traj=4, lambda_ = lambda_, ignore_discriminator=0, device='cuda', title = discr_str, figsize = 8, logit=False,
                      random_state=42,
                      start_from_origin=True)
