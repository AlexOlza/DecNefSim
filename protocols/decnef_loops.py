#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:39:45 2025

@author: alexolza
"""

import torch
from matplotlib import pyplot as plt
from components.generators import  get_data_predictions, get_classes_mean
import numpy as np
import random
from scipy.spatial.distance import euclidean
from visualization.plotting import show_image, show_images_grid, plot_vae2d_trajectory, plot_vae2d_random_trajectory
import matplotlib.cm as cm
from matplotlib import colors as cplt
from components.update_rules import fp0
 
def minimal_loop(train_loader, generator, discriminator, target_class, lambda_, n_iter, device,  
                 update_rule_func, p_scale_func=fp0,#identity_f_p, 
                 z_current=None, ignore_discriminator = 0, random_state=0, noise_sigma=torch.tensor(1.0),
                 warm_up = 5, stop_eps=1e-3,
                 **update_rule_kwargs):
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.cuda.manual_seed(random_state)
    z_dim = generator.z_dim
    idx = 0 if target_class==min(discriminator.classes) else 1
    if z_current is None:
        z_current = torch.zeros(1,z_dim) # generates blurry image, which is the average of all seen data

    X0 = generator.decoder(z_current.to(device), generator.target_size)
 
    if ignore_discriminator:
        p = torch.rand(1)
        d_str = 'Random p'
    else:
        d_str = 'Discr.'
        with torch.no_grad():
            p =  torch.nn.Softmax()(discriminator(X0).flatten())[idx]
    generated_images=[X0[0].cpu()]
    probabilities = [p.item()]
    all_probabilities = [p.item()]
    # p0 = p.item()#+np.random.uniform(1e-4,1e-3)# abs(p.item()-1e-6)
    trajectory =[z_current.numpy().flatten()]
    D = []
    sigmas = [noise_sigma]
    past_probabilities_mean = p.to(device)
    recent_probabilities_mean = p.to(device)
    patience0 = 25
    patience=patience0
    for i in range(1, n_iter+1):
        if i>=150: 
            if (recent_probabilities_mean.item()>= 0.9) or sigmas[-1]<0.1:
                patience-=1
                print(f'iter {i}, patience: ', patience)
            else:
                patience = patience0
        if patience==0: break
        if i>1:
            recent_probabilities_mean = torch.tensor(np.mean(all_probabilities[-i*warm_up:])).to(device)
            past_probabilities_mean = torch.tensor(np.mean(all_probabilities[-(i+1)*warm_up: -i*warm_up])).to(device)
            # print(past_probabilities_mean, recent_probabilities_mean, sigmas[-1], z_current)
        for j in range(2*warm_up):   
            z_new, _ = update_rule_func(z_current, recent_probabilities_mean, lambda_, p_scale_func, p0=past_probabilities_mean, 
                                        noise_sigma_0 = sigmas[-1].to(device), sigma0=sigmas[-1], 
                                        **update_rule_kwargs)
            # z_new = update_rule_func(z_current, p, target_dist, lambda_, p_scale_func, p0=probabilities[0], **update_rule_kwargs)
            # z_new, h_icdf = update_latent(z, mean, cov, p.cpu().item(), lambda_val, np.pi/2, random_noise)
            # sigmas.append(noise_sigma)
            z_new = torch.tensor(z_new).float()
            # z_new = (1 - lambda_) * z_current + lambda_ * h_icdf
            x_decoded = generator.decoder(z_new.to(device), generator.target_size)
            z_current = z_new

            if ignore_discriminator==0:
                with torch.no_grad():
                    p =  torch.nn.Softmax()(discriminator(x_decoded).flatten())[idx]
            else: 
                p = torch.rand(1)
            
            all_probabilities.append(p.item())
            
            
        with torch.no_grad():
            if noise_sigma > 1e-2: 
                z_new, noise_sigma = update_rule_func(z_current, recent_probabilities_mean, lambda_, p_scale_func, p0=past_probabilities_mean, 
                                                      noise_sigma_0 = sigmas[-1].to(device), sigma0=sigmas[-1], 
                                                      **update_rule_kwargs)
            # z_new = update_rule_func(z_current, p, target_dist, lambda_, p_scale_func, p0=probabilities[0], **update_rule_kwargs)
            # z_new, h_icdf = update_latent(z, mean, cov, p.cpu().item(), lambda_val, np.pi/2, random_noise)
            sigmas.append(noise_sigma)
            # print(sigmas[-2:])
            z_new = torch.tensor(z_new).float()
            # z_new = (1 - lambda_) * z_current + lambda_ * h_icdf
            x_decoded = generator.decoder(z_new.to(device), generator.target_size)
            
            if ignore_discriminator:
                p = torch.rand(1)
            else:
                with torch.no_grad():
                    p =  torch.nn.Softmax()(discriminator(x_decoded).flatten())[idx]

            all_probabilities.append(p.item())
            generated_images.append(x_decoded[0].cpu())
            z_current = z_new
            trajectory.append(z_new.cpu().numpy().flatten())
            
        probabilities.append(recent_probabilities_mean.item())  
    Dnorm = np.array(D)#/D[0]
    sigmas = np.array(sigmas)
    return  generated_images, np.array(trajectory), Dnorm, np.array(probabilities), np.array(all_probabilities), sigmas
#%%
def compute_single_trajectory(vae, discriminator, trajectory_random_seed,
             train_loader, target_class, update_rule_func, p_scale_func, trajectory_name, 
                                   n_iter = 15, lambda_ = 0.15, device='cuda', ignore_discriminator=0,
                                   start_from_origin=True,
                                   **f_p_kwargs): 
    torch.manual_seed(trajectory_random_seed)
    random.seed(trajectory_random_seed)
    np.random.seed(trajectory_random_seed)
    torch.cuda.manual_seed(trajectory_random_seed)
    z_current = None if start_from_origin else torch.normal(0.,1.,(1, 2))#starting_zs[iter_]
    generated_images, trajectory, probabilities, all_probabilities, sigma  = minimal_loop(train_loader, vae, discriminator, target_class, lambda_, n_iter, device,
                                                                                          update_rule_func, p_scale_func,z_current, ignore_discriminator)
    return generated_images, trajectory, probabilities, all_probabilities, sigma

def show_trajectories(vae_2d, discriminator, train_loader, target_class, class_names, update_rule_func, p_scale_func, filename, extension='png',
                      n_iter = 15, n_traj=10, lambda_ = 0.15, device='cuda', ignore_discriminator=0, title = '', figsize = 8, figsize_gaussians= 12,
                      logit=False, random_state=0, zoom_radius = 1.5, start_from_origin=False,
                      # starting_zs=None,
                      **f_p_kwargs):
    latents_mean, latents_stdvar, labels = get_data_predictions(vae_2d, train_loader)
    classes_mean = get_classes_mean(train_loader, labels, latents_mean, latents_stdvar) 
    trajectories = []
    generated_images = {}
    probabilities = {}
    all_probabilities = {}
    Dnorm = {}
    sigma={}
    # starting_zs = [torch.normal(0.,1.,(1, 2)) for i in range(n_traj)] 
    colors = [cplt.to_hex(cm.tab10(i)) for i in range(n_traj)]
    idxs =[3, 4, 6, 9] 
    
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.cuda.manual_seed(random_state)
    
    idxs = idxs if all([c < n_traj for c in idxs]) else sorted(np.random.choice(range(n_traj),size=4,replace=False))
    for iter_ in range(n_traj):
        z_current = None if start_from_origin else torch.normal(0.,1.,(1, 2))#starting_zs[iter_]
        generated_images[iter_], trajectory, Dnorm[iter_], probabilities[iter_], all_probabilities[iter_], sigma[iter_]  = minimal_loop(train_loader, vae_2d, discriminator, target_class, lambda_, n_iter, device,
                                                                                                                                    **f_p_kwargs)
        trajectories.append(trajectory)
    return trajectories, Dnorm, probabilities, sigma
    
def plot_trajectories(vae_2d,
                      trajectories, generated_images, Dnorm, probabilities, sigma,
                      train_loader, target_class, class_names,
                      figsize, figsize_gaussians, zoom_radius,
                      ignore_discriminator,
                      mark_origin,
                      title, filename, extension):
    fig_traj1, axs1, fig_traj2, axs2, colors = plot_vae2d_trajectory(vae_2d, trajectories, train_loader, target_class, class_names, 
                                                  figsize_gaussians, zoom_radius = zoom_radius, mark_origin= mark_origin)
    # axs1.axis('off')
    axs2.axis('off')
    fig_traj1.tight_layout()
    fig_traj2.tight_layout()
    fig_traj2.savefig(f'{filename}_trajectories.{extension}',  format=extension)
    fig_traj1.savefig(f'{filename}_trajectories_zoomed.{extension}',format=extension)

    fig3 = plt.figure(figsize = (4*figsize, 2.4*figsize),constrained_layout=False)
    # fig3.suptitle(discr_str, fontsize=20, weight='bold')
    gs = fig3.add_gridspec(3,4)
    # ax_gauss = fig3.add_subplot(gs[0, :])
    ax_1 = fig3.add_subplot(gs[0, 0])
    ax_2 = fig3.add_subplot(gs[0, 1])
    ax_3 = fig3.add_subplot(gs[0, 2])
    ax_4 = fig3.add_subplot(gs[0, 3])
    ax_21 = fig3.add_subplot(gs[1, 0])
    ax_22 = fig3.add_subplot(gs[1, 1])
    ax_23 = fig3.add_subplot(gs[1, 2])
    ax_24 = fig3.add_subplot(gs[1, 3])
    ax_31 = fig3.add_subplot(gs[2, 0])
    ax_32 = fig3.add_subplot(gs[2, 1])
    ax_33 = fig3.add_subplot(gs[2, 2])
    ax_34 = fig3.add_subplot(gs[2, 3])
 
    row2 = [ax_1, ax_2, ax_3, ax_4]
    row3 = [ax_21, ax_22, ax_23, ax_24]
    row5 = [ax_31, ax_32, ax_33, ax_34]
    
    xticks = {}
    for idx in range(len(Dnorm)):
        length = len(Dnorm[idx])
        xticks[idx] = np.floor(np.linspace(0, length, 5)).astype(int) if length >= 5 else range(length+1)
    for i, a in enumerate(row5):
        
        # a.plot(Dnorm[idxs[i]])
        a.plot(Dnorm[i])
        a.axhline(y=Dnorm[i][0], color='grey', ls='--')
        a.set_title('Distance', color = colors[i], fontsize = 38)
        a.set_xlabel('Iterations', color = colors[i], fontsize = 36)
        # if logit: a.set_yscale('logit')
        a.set_xticks(xticks[i])
        a.set_xticklabels(xticks[i], fontsize = 28)
        for spine in a.spines.values():
                spine.set_visible(True)  # Ensure spines are not hidden
    
    for i, a in enumerate(row3):
        
        a.plot(probabilities[i], color ='tab:red', label='Probability')
        a.set_ylabel('Probability', color='tab:red', fontsize = 28)
        a.tick_params(axis='y', labelcolor='tab:red', size = 28, labelsize = 28)

        # a.set_title(axtitle, color = colors[idxs[i]], fontsize = 24)
        # a.set_xlabel('Iterations', color = colors[idxs[i]], fontsize = 18)
        # if logit: a.set_yscale('logit')
        ax2 = a.twinx()
        ax2.set_ylabel('Variance', color='tab:blue', fontsize = 30)
        ax2.tick_params(axis='y', labelcolor='tab:blue', size = 28, labelsize = 28)

        ax2.plot(sigma[i], label = 'sigma')
        # a.set_title(f'p_start={probabilities[i][0]:.3f}, p_end={probabilities[i][-1]:.3f}', color = colors[i], fontsize = 20)
        a.set_xlabel('Iterations', color = colors[i], fontsize = 30)
        a.axis('tight')
        a.set_xticks(xticks[i])
        a.set_xticklabels(xticks[i], fontsize = 28)
        a.tick_params(axis='y', size = 28, labelsize = 28)
        for spine in a.spines.values():
                spine.set_visible(True)  # Ensure spines are not hidden
        # a.legend()
    ax_34 = fig3.add_subplot(gs[2, 3])  # Re-add it
    # a.plot(probabilities[idxs[3]])
    # ax_34.set_title('Variance', color = colors[idxs[3]], fontsize = 24)
    ax_34.set_xlabel('Iterations', color = colors[3], fontsize = 30)
    ax_34.set_xticks(xticks[i])
    ax_34.set_xticklabels(xticks[i], fontsize = 28)
    ax_34.set_frame_on(True)
    ax_34.set_visible(True)  # Ensures the axis is visible
    ax_34.xaxis.set_visible(True)
    ax_34.yaxis.set_visible(True)
    ax_34.spines['top'].set_visible(True)
    ax_34.spines['right'].set_visible(True)
    ax_34.spines['bottom'].set_visible(True)
    ax_34.spines['left'].set_visible(True)
    
    for i, a in enumerate(row2):
        length = len(Dnorm[i])
        if length >= 50: 
            chosen = np.floor(np.linspace(0, length, 49)).astype(int)
            imgs = [generated_images[i][idx] for idx in chosen]
            length = 49
        else: 
            imgs = generated_images[i]
        show_images_grid(imgs,np.floor(np.sqrt(length+1)).astype(int),a, title='')
        if length==49:
            aa = a.twinx()
            # a.yaxis.tick_right()
            aa.set_yticks(range(np.floor(np.sqrt(length+1)).astype(int)))
            aa.tick_params(axis='y', labelcolor=colors[i])
            aa.set_yticklabels(chosen[6::7][::-1], fontsize = 26)
            a.set_frame_on(False)
            a.get_xaxis().set_visible(False)
            aa.axis('off')
            # a.get_xaxis().set_visible(False)
        else:
            a.axis('off')
            a.yaxis.tick_right()
            a.set_yticks(range(np.floor(np.sqrt(length+1)).astype(int)))
            a.tick_params(axis='y', labelcolor=colors[i])
            a.set_yticklabels(chosen[6::7][::-1], fontsize = 36)
            a.set_frame_on(False)
            a.get_xaxis().set_visible(False)
        a.set_title(f'Trajectory {i+1}', color = colors[i], fontsize = 38)
    
    if len(title)>0: fig3.suptitle(title, y=1, fontsize=42)
    # fig3.canvas.draw_idle()  
    fig3.tight_layout()
    fig3.savefig(f'{filename}.{extension}', dpi=300, format=extension)
    
    if ignore_discriminator:
        fig_r, ax_r = plt.subplots(figsize = (15,6),layout='constrained')
        
        ax_r = plot_vae2d_random_trajectory(vae_2d, trajectories, train_loader, target_class, class_names, ax_r,
                                  figsize=10)
        
        fig_r.savefig(f'{filename}_random_trajs.{extension}', dpi=300, format=extension)
        fig_r.savefig(f'{filename}_random_trajs.png', transparent=True, dpi=300, format=extension)

"""
Passing a zero latent vector (all zeros) and passing a random latent vector (sampled from the prior) yield different outputs because of where they lie in the latent space and how the decoder has learned to interpret them:

- Zeros as the Latent Code:   
  - During training, the VAE learns to map the diverse features of the training data to a distribution (usually N(0,1)). The center of this distribution is an “average” of all latent representations.  
  - When you pass a zero vector, the decoder produces a reconstruction that is an average over all possible images it has seen, resulting in a blurry image.

- Random Numbers as the Latent Code:  
  - Random latent vectors are drawn from the prior distribution and therefore capture the variability of the latent space.  
  - These vectors correspond to specific points on the learned manifold, which the decoder has been trained to map to realistic reconstructions.  
  - As a result, the output image will be plausible and detailed

Zeros lead to blurry images because they force the decoder to generate an “average” output, while random samples yield plausible images because they reflect the learned features of the training distribution.
"""
#%%
def minimal_loop_legacy(train_loader, generator, discriminator, target_class, lambda_, n_iter, device,  
                 class_names, classes_mean, update_rule_func, p_scale_func=fp0,#identity_f_p, 
                 z_current=None, ignore_discriminator = 0, random_state=0, noise_sigma=torch.tensor(1.0),
                 title='', title_color = 'black', n_str = '', plot=False, warm_up = 5, stop_eps=1e-3,
                 **update_rule_kwargs):
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.cuda.manual_seed(random_state)
    z_dim = generator.z_dim
    if z_current is None:
        z_current = torch.zeros(1,z_dim) # generates blurry image, which is the average of all seen data
    # elif mode=='normal':
    #     # z_current= torch.randn(1, z_dim) # generates randomly among the classes seen during training
    #     z_current= torch.normal(0.,1.,(1, z_dim)) # generates randomly among the classes seen during training
    # else:
    #     z_current = -2 * torch.rand((1, z_dim)) + 1
    X0 = generator.decoder(z_current.to(device), generator.target_size)
    # show_image(X0.cpu(), title=f'Initial random image')
    idx = 0 if target_class==min(discriminator.classes) else 1
    noidx = 0 if idx==1 else 1
    non_target_class = list(set(discriminator.classes)-set([target_class]))[0]
    target_class_name, non_target_class_name = class_names[idx], class_names[noidx]
    latents_mean_target, latents_stddev_target = classes_mean[target_class]
    
    if ignore_discriminator==3:
        p = torch.tensor(1e-3)
        d_str = 'Det. growing p'
    elif ignore_discriminator==2:
        p = torch.rand(1)
        d_str = 'Random p'
    elif ignore_discriminator==1:
        d_str = 'p = exp(-dist)'
        with torch.no_grad():
            p = torch.exp(- (z_current - latents_mean_target).pow(2).sum().sqrt()).to(device)
    else:
        d_str = 'Discr.'
        with torch.no_grad():
            p =  torch.nn.Softmax()(discriminator(X0).flatten())[idx]
    generated_images=[X0[0].cpu()]
    probabilities = [p.item()]
    all_probabilities = [p.item()]
    # p0 = p.item()#+np.random.uniform(1e-4,1e-3)# abs(p.item()-1e-6)
    trajectory =[z_current.numpy().flatten()]
    D = []
    sigmas = [noise_sigma]
    # target_dist = Normal(latents_mean_target, latents_stddev_target)
    # target_dist = MultivariateNormal(latents_mean_target, covariance_matrix=latents_stddev_target*torch.eye(z_dim))
    target_prototype = generator.decoder(latents_mean_target.to(device), generator.target_size)
    non_target_prototype = generator.decoder(classes_mean[non_target_class][0].to(device), generator.target_size)
    # noise_sigma_0=noise_sigma
    
    past_probabilities_mean = p.to(device)
    recent_probabilities_mean = p.to(device)
    patience0 = 25
    patience=patience0
    # print(noise_sigma)
    for i in range(1, n_iter+1):
        if i>=150: 
            if (recent_probabilities_mean.item()>= 0.9) or sigmas[-1]<0.1:
                patience-=1
                print(f'iter {i}, patience: ', patience)
            else:
                patience = patience0
        if patience==0: break
        if i>1:
            recent_probabilities_mean = torch.tensor(np.mean(all_probabilities[-i*warm_up:])).to(device)
            past_probabilities_mean = torch.tensor(np.mean(all_probabilities[-(i+1)*warm_up: -i*warm_up])).to(device)
            # print(past_probabilities_mean, recent_probabilities_mean, sigmas[-1], z_current)
        for j in range(2*warm_up):   
            z_new, _ = update_rule_func(z_current, recent_probabilities_mean, lambda_, p_scale_func, p0=past_probabilities_mean, 
                                        noise_sigma_0 = sigmas[-1].to(device), sigma0=sigmas[-1], 
                                        **update_rule_kwargs)
            # z_new = update_rule_func(z_current, p, target_dist, lambda_, p_scale_func, p0=probabilities[0], **update_rule_kwargs)
            # z_new, h_icdf = update_latent(z, mean, cov, p.cpu().item(), lambda_val, np.pi/2, random_noise)
            # sigmas.append(noise_sigma)
            z_new = torch.tensor(z_new).float()
            # z_new = (1 - lambda_) * z_current + lambda_ * h_icdf
            x_decoded = generator.decoder(z_new.to(device), generator.target_size)
            z_current = z_new

            if ignore_discriminator==0:
                with torch.no_grad():
                    p =  torch.nn.Softmax()(discriminator(x_decoded).flatten())[idx]
            else: 
                p = torch.rand(1)
            
            all_probabilities.append(p.item())
            
            
        with torch.no_grad():
            if noise_sigma > 1e-2: 
                z_new, noise_sigma = update_rule_func(z_current, recent_probabilities_mean, lambda_, p_scale_func, p0=past_probabilities_mean, 
                                                      noise_sigma_0 = sigmas[-1].to(device), sigma0=sigmas[-1], 
                                                      **update_rule_kwargs)
            # z_new = update_rule_func(z_current, p, target_dist, lambda_, p_scale_func, p0=probabilities[0], **update_rule_kwargs)
            # z_new, h_icdf = update_latent(z, mean, cov, p.cpu().item(), lambda_val, np.pi/2, random_noise)
            sigmas.append(noise_sigma)
            # print(sigmas[-2:])
            z_new = torch.tensor(z_new).float()
            # z_new = (1 - lambda_) * z_current + lambda_ * h_icdf
            x_decoded = generator.decoder(z_new.to(device), generator.target_size)
            
            if ignore_discriminator==3:
                p = torch.tensor(np.linspace(1e-3,1-1e-3,n_iter+1)[i+1])
            elif ignore_discriminator==2:
                p = torch.rand(1)
            elif ignore_discriminator==1:
                with torch.no_grad():
                    p = torch.exp(- (z_current.to(device) - latents_mean_target.to(device)).pow(2).sum().sqrt())
            else:
                with torch.no_grad():
                    p =  torch.nn.Softmax()(discriminator(x_decoded).flatten())[idx]

            all_probabilities.append(p.item())
            
            # H.append(h.item())
            D.append(euclidean(z_new.cpu().numpy().flatten(),latents_mean_target.flatten()))
            # show_image(x_decoded[0].cpu(), title=f'i={i}, p={p.item():.6f}')
            generated_images.append(x_decoded[0].cpu())
            z_current = z_new
            trajectory.append(z_new.cpu().numpy().flatten())
            
        probabilities.append(recent_probabilities_mean.item())  
    Dnorm = np.array(D)#/D[0]
    sigmas = np.array(sigmas)
    if plot:
        fig3 = plt.figure(constrained_layout=True)
        gs = fig3.add_gridspec(2,2)
        ax_tgt = fig3.add_subplot(gs[0, 0])
        ax_nontgt = fig3.add_subplot(gs[1, 0])
        ax_gen = fig3.add_subplot(gs[:,1])
        ax_tgt=show_image(target_prototype[0].detach().cpu(),ax_tgt, title=f'Target prototype ({target_class_name})')
        ax_nontgt=show_image(non_target_prototype[0].detach().cpu(),ax_nontgt, title=f'Non target prototype ({non_target_class_name})')
        plt.axis('off')
        ax_gen=show_images_grid(generated_images,np.floor(np.sqrt(n_iter+1)).astype(int),ax_gen, title=f'{d_str}, {n_str} (zdim={z_dim})')
        # plt.show()
        fig, axs = plt.subplots(1,2, figsize = (30, 10))
        axes=axs.flatten()
        axes[0].plot(range(n_iter+1),np.array(probabilities).flatten())
        axes[0].set_title(f'p(y=0|X) (zdim={z_dim})')
        # axes[1].plot(range(n_iter),np.array(H).flatten())
        # axes[1].set_title(f'h(p)')
        axes[1].plot(range(n_iter),Dnorm.flatten())
        axes[1].set_title(f'Distance from X to the {target_class_name} prototype (zdim={z_dim})')
        nrows = np.ceil(np.sqrt(n_iter+1)).astype(int)
        ncols = np.floor(np.sqrt(n_iter+1)).astype(int)
        fig_gen, axxx = plt.subplots(nrows, ncols,figsize = (nrows*2,ncols*2))
        axx = axxx.flatten()
        for ax,img,p_,d_ in zip(axx, generated_images, probabilities, Dnorm): 
            x, y = img[0].detach().numpy().shape
            ax.imshow(img[0].detach().numpy())
            ax.text(x/2, (y/2)+2,f'p={p_:.2e}', color='red',
                  bbox={'facecolor':'white','alpha':0.2,'edgecolor':'none','pad':1},
                  ha='center', va='center') 
            ax.text(x/2, (y/2)-2,f'd={d_:.2e}', color='red',
                  bbox={'facecolor':'white','alpha':0.2,'edgecolor':'none','pad':1},
                  ha='center', va='center') 
            ax.axis('off')
        for ax in axx[n_iter:]:
            ax.axis('off')
    return  generated_images, np.array(trajectory), Dnorm, np.array(probabilities), np.array(all_probabilities), sigmas
