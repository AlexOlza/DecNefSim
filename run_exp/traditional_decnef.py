#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 15:18:45 2025

@author: alexolza
"""
import sys
sys.path.append('.')
import torch
from tqdm import tqdm
import os
import regex as re
import numpy as np
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
###########################
from components.generators import train_model, get_data_predictions, get_classes_mean
from visualization.plotting import visual_eval_vae
from protocols.decnef_loops import minimal_loop, show_trajectories, update_z_fixed_normal_drift, identity_f_p
from components.discriminators import CNNClassification, BinaryDataLoader, plot_predicted_probability_distribution
from components.generators import dist_classmean_to_random
from components.update_rules import update_z_moving_normal_drift_adaptive_variance, powsig
#%%
"""
Configuration variables
"""
EXP_NAME = 'trash'#sys.argv[1]
z_dim = 2
lambda_ = 0.025
#%%

figpath = f'../EXPERIMENTS/{EXP_NAME}/figures/'
outpath = f'../EXPERIMENTS/{EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{EXP_NAME}/weights/'

genfigpath = figpath+'generator_eval'
disfigpath = figpath+'discriminator_eval'
nfbfigpath = figpath+'nfb_eval'
nfb_traj_figpath = os.path.join(nfbfigpath,'VAE2D_trajectories')
dis_pdist_figpath = os.path.join(disfigpath,'probabilities_by_classifier') 
reconstruction_path = outpath + 'reconstructions'

for p in [genfigpath, dis_pdist_figpath,
          nfb_traj_figpath, reconstruction_path,
          modelpath]:
    if not os.path.exists(p):
        os.makedirs(p)
#%%
device='cuda'
batch_size=64
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the MNIST training data
trainset = datasets.FashionMNIST('.', download=True, train=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the MNIST test data
testset = datasets.FashionMNIST('.', download=True, train=False, transform=transform)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)

if hasattr(trainset, 'class_to_idx'):
    class_name_dict = trainset.class_to_idx
else:
    class_name_dict = {v: v for v in trainset.classes}
    
class_names, class_numbers = np.array([[k,v] for k,v in class_name_dict.items()]).T
class_numbers = class_numbers.astype(int)
img_size = trainset[0][0].shape[-1]
#%%
vae, vae_history = train_model(train_loader,batch_size, epochs=20, z_dim=z_dim)
print('GENERATOR TRAINING FINISHED WITH z_dim=',z_dim)
#%%
eval_figs = visual_eval_vae(vae, vae_history, z_dim, train_loader, class_names, class_numbers)
ext = 'png'
fignames = [f'{name}{z_dim}.{ext}' for name in ['LOSS','REC','PROT','LATENT_VIS']]
for figure, figurename in zip(eval_figs, fignames):
    figure.savefig(os.path.join(genfigpath, figurename))
#%%
vae_2d, vae_2d_history = train_model(train_loader, batch_size, epochs=20, z_dim=2)
print('GENERATOR TRAINING FINISHED WITH z_dim=',2)
#%%

eval_figs2 = visual_eval_vae(vae_2d, vae_2d_history, 2, train_loader, class_names, class_numbers)
fignames = [f'{name}2.{ext}' for name in ['LOSS','REC','PROT','LATENT_VIS', 'LATENT_TRAV']]
for figure, figurename in zip(eval_figs2, fignames):
    figure.savefig(os.path.join(genfigpath, figurename), format="png")

#%%


def train_all_discriminators(train_loader, classes=None, class_names=None, discr_epochs=10):
    discr_dict = {}
    if classes is None: classes = trainset.targets.unique().numpy()
    if class_names is None: class_names = trainset.classes
    class_combinations = list(itertools.combinations(classes, 2))
    
    for combo in tqdm(class_combinations):
        combo_names = [list(class_names)[i] for i in combo]
        discr_str = f'{combo_names[0]} vs {combo_names[1]}'
        print('DISCRIMINATOR TRAINING: ',discr_str)
        tl = BinaryDataLoader(trainset, list(combo), batch_size=16)
        testl = BinaryDataLoader(testset, list(combo), batch_size=16)
        discriminator = CNNClassification(torch.Size([1, 28, 28]), combo, device, name=discr_str)  # trainset[0][0].shape 
        discriminator.evaluate(testl)
        discriminator.fit( epochs=discr_epochs, lr=1e-3, train_loader=tl, val_loader = testl)
        discr_dict[discr_str] = discriminator
    
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
    
        # # Convert Figure objects to images.
        # img1, img2 = fig_to_image(fig_gen[0]), fig_to_image(fig_gen[1])
        # img3, img4 = fig_to_image(fig_gen2d[0]), fig_to_image(fig_gen2d[1])
        
        # # Create a new figure to display the images side by side.
        # fig_combined, axs = plt.subplots(2, 2, figsize = (30,30))
        # axs[0][0].imshow(img1)
        # axs[0][0].axis('off')
        # axs[0][1].imshow(img2)
        # axs[0][1].axis('off')
        # axs[1][0].imshow(img3)
        # axs[1][0].axis('off')
        # axs[1][1].imshow(img4)
        # axs[1][1].axis('off')
        # plt.tight_layout()
        # fig_combined.savefig(figname_nfb)
        # plt.show()
    


#%% TRAINING ALL DISCRIMINATORS
discr_epochs = 10
discr_dict = train_all_discriminators(train_loader, classes=None, class_names=None, discr_epochs=discr_epochs)

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
slope=10
#%%
generated_images, trajectory, Dnorm, probabilities =  full_clasewise_DecNef(vae, discr_dict, train_loader, lambda_, n_iter, 
                                                                            classes=None, class_names=None, reverse = False, z_current = None,
                                                                            update_rule_func=update_z_fixed_normal_drift, p_scale_func=identity_f_p,
                                                                            device='cuda',
                                                                            random_state=0,
                                                                            )

#%%
generated_images_2d, trajectory_2d, Dnorm_2d, probabilities_2d =  full_clasewise_DecNef(vae_2d, discr_dict, train_loader, lambda_, n_iter, 
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
    show_trajectories(vae_2d, discr_dict[discr_str], train_loader, target_class, class_names,
                      update_z_moving_normal_drift_adaptive_variance,#update_z_fixed_normal_drift,
                      p_scale_func = powsig,#linsig,#f,#identity_f_p,
                      noise_sigma=torch.tensor(1.0),# starting_zs = None,
                      
                      filename = os.path.join(nfb_traj_figpath,f'{str_}_conv'),extension = 'pdf', zoom_radius=1.2,
                      n_iter = 600, n_traj=4, lambda_ = lambda_, ignore_discriminator=0, device='cuda', title = discr_str, figsize = 8, logit=False,
                      random_state=42,
                      start_from_origin=True)
