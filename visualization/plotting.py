#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:16:47 2025

@author: alexolza
"""

# from torch import nn
import torch
# from tqdm import tqdm
from torch.distributions.normal import Normal
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from umap.umap_ import UMAP
# from torch.distributions.kl import kl_divergence
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.cm as cm
from matplotlib import colors as cplt
import io
from PIL import Image
from scipy.stats import multivariate_normal
from components.generators import get_data_predictions, get_classes_mean

def show_images_grid(images, class_num, ax, title, title_color='black', text=[], title_fontsize=26):
    grid = make_grid(images, nrow=class_num, normalize=True).permute(1,2,0).numpy()
    ax.imshow(grid)
    ax.set_title(title, color = title_color, fontsize=title_fontsize)
    plt.axis('off')
    # _ = plt.yticks([])
    return ax
def show_image(image, ax, title=''):
    ax.imshow(image.permute(1,2,0).numpy())
    ax.set_title(title)
    plt.axis('off')
    # _ = plt.yticks([])
    return ax

def traverse_two_latent_dimensions(model, input_sample, z_dist, n_samples=25, z_dim=16, dim_1=0, dim_2=1, title='plot', device='cuda',digit_size=28):
  

  percentiles = torch.linspace(1e-6, 0.9, n_samples)

  grid_x = z_dist.icdf(percentiles[:, None].repeat(1, z_dim))
  grid_y = z_dist.icdf(percentiles[:, None].repeat(1, z_dim))

  figure = np.zeros((digit_size * n_samples, digit_size * n_samples))

  z_sample_def = input_sample.clone().detach()
  target_size = torch.Size([digit_size, digit_size])
  # select two dimensions to vary (dim_1 and dim_2) and keep the rest fixed
  for yi in range(n_samples):
      for xi in range(n_samples):
          with torch.no_grad():
              z_sample = z_sample_def.clone().detach()
              z_sample[:, dim_1] = grid_x[xi, dim_1]
              z_sample[:, dim_2] = grid_y[yi, dim_2]
              x_decoded = model.decoder(z_sample.to(device),target_size).cpu()
              # print(x_decoded.shape)
          digit = x_decoded[0].reshape(digit_size, digit_size)
          figure[yi * digit_size: (yi + 1) * digit_size,
                 xi * digit_size: (xi + 1) * digit_size] = digit.numpy()

  fig, ax = plt.subplots(figsize=(15,15))
  ax.imshow(figure, cmap='Greys_r')
  plt.axis('off')
  # ax.set_xticks([round(digit_size*i,3) for i in range(n_samples)],percentiles.numpy(), rotation=45)
  # ax.set_yticks([round(digit_size*i,3) for i in range(n_samples)],percentiles.numpy())
  plt.title(title)
  # plt.show() 
  return fig

def visualize_latent_space(model, data_loader, class_names, device, method='TSNE', num_samples=10000):
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
          if len(latents) > num_samples:
            break
          mu, _ = model.encoder(data.to(device))
          latents.append(mu.cpu())
          labels.append(label.cpu())

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    # print(latents.shape)
    # print(labels.shape)
    assert method in ['TSNE', 'UMAP'], 'method should be TSNE or UMAP'
    if method == 'TSNE':
        tsne = TSNE(n_components=2, verbose=1)
        tsne_results = tsne.fit_transform(latents)
        df = pd.DataFrame()
        df['TSNE-1'] = tsne_results[:, 0]
        df['TSNE-2'] = tsne_results[:, 1]
        df['Class'] = [list(class_names)[i] for i in labels.astype(int)]
        df = df.sort_values('Class')
        fig_data, ax_data = plt.subplots(1, figsize = (20, 10))
        sns.scatterplot(
                    x="TSNE-1", y="TSNE-2",
                    hue='Class',  # coloring
                    data=df,
                    alpha=1, ax=ax_data
                )
        ax_data.set_title(f'{latents.shape[-1]}-D VAE Latent Space with TSNE',fontsize= 22)  
    elif method == 'UMAP':
        reducer = UMAP()
        embedding = reducer.fit_transform(latents)
        df = pd.DataFrame()
        df['UMAP-1'] = embedding[:, 0]
        df['UMAP-2'] = embedding[:, 1]
        df['Class'] = [list(class_names)[i] for i in labels.astype(int)]
        df = df.sort_values('Class')
        fig_data, ax_data = plt.subplots(1, figsize = (20, 10))
        sns.scatterplot(
                    x="UMAP-1", y="UMAP-2",
                    hue='Class',  # coloring
                    data=df,
                    alpha=1, ax=ax_data, palette = sns.color_palette("tab10")
                )
        ax_data.set_title(f'{latents.shape[-1]}-D VAE Latent Space with UMAP ',fontsize= 22)  
    plt.show()
    return fig_data

def visual_eval_vae(vae, vae_history, z_dim, train_loader, class_names, class_numbers, device='cuda'):
    fig_loss, ax = plt.subplots(1,2)
    bce = vae_history.train_BCE.plot(title = f'BCE loss (z_dim = {z_dim})', ax = ax[0])
    kl = vae_history.train_KL.plot(title = f'KL divergence (z_dim = {z_dim})',ax=ax[1])
    i=0
    rec, ax = plt.subplots(2, 3)
    for img, lbl in train_loader:
        img = img[0][0] 
        ax[0][i].imshow(img, cmap='gray') 
        ax[1][i].imshow(vae(img.unsqueeze(0).unsqueeze(0).to(device))[0].detach().cpu()[0,0], cmap='gray')
        ax[0][i].set_title(lbl[0].item())
        
        i+=1
        if i>=3: break
    plt.axis('off')
    rec.suptitle(f'Reconstruction (z_dim={z_dim})')
    # rec.show()
    
    latents_mean, latents_stdvar, labels = get_data_predictions(vae, train_loader)
    classes_mean = get_classes_mean(train_loader, labels, latents_mean, latents_stdvar)

    prot, axs = plt.subplots(1, len(class_numbers), figsize=(4*len(class_numbers),4))
    axss = axs.flatten()
    i=0

        
    for label, idx in zip(class_names, class_numbers):
        latents_mean_target, latents_stddev_target = classes_mean[int(idx)]
        target_prototype = vae.decoder(torch.Tensor(latents_mean_target).to(device), vae.target_size)
        axss[i]=show_image(target_prototype[0].detach().cpu(),axss[i], title=f'{label}')
        i+=1
    plt.axis('off')
    prot.suptitle(f'Class prototypes (z_dim={z_dim})')
    # plt.show()   
    
    latent_vis = visualize_latent_space(vae, train_loader, class_names,
                           device='cuda' if torch.cuda.is_available() else 'cpu',
                           method='UMAP', num_samples=10000)
    
    z_dist = Normal(torch.zeros(1, 2), torch.ones(1, 2))
    input_sample = torch.zeros(1, 2)
    if z_dim==2:
        latent_trav = traverse_two_latent_dimensions(vae, input_sample, z_dist, n_samples=20, dim_1=0, dim_2=1, z_dim=2, title='Traversing 2D latent space', device=device)
    else:
        latent_trav = plt.figure()
    return fig_loss, rec, prot, latent_vis, latent_trav

def plot_gaussians(classes_mean, class_names, target_class, cmaps, ax, zoomed=False, zoom_radio=1, labels=True):
    xlim, ylim = [], []
    tgt_mu, _  = classes_mean[target_class]
    # tgt_mu = tgt_mu.numpy().flatten()
    tgt_mu = tgt_mu.flatten()
    zoomed_xmin, zoomed_xmax = tgt_mu[0]-zoom_radio, tgt_mu[0]+zoom_radio
    zoomed_ymin, zoomed_ymax = tgt_mu[1]-zoom_radio, tgt_mu[1]+zoom_radio
    for c in range(10):
        mu, sigma = classes_mean[c]
        mu = mu.flatten()
        sigma = sigma.flatten()
        # mu = mu.numpy().flatten()
        # sigma = sigma.numpy().flatten()
        # Initializing the covariance matrix
        cov = np.diag(sigma)
        x, y = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
        
        pos = np.dstack((x, y))
        rv = multivariate_normal(mu, cov)
        pdf_values = rv.pdf(pos)
        pdf_masked = np.ma.masked_where(pdf_values < 0.05, pdf_values)
        mask = pdf_values > 0.05
        x_min, x_max = x[mask].min(), x[mask].max()
        y_min, y_max = y[mask].min(), y[mask].max()
        xlim.append([x_min, x_max])
        ylim.append([y_min, y_max])
        
        # Plot contour with a unique colormap
        contour = ax.contourf(x, y, pdf_masked, levels=15, cmap=cmaps[c])
        alpha_values = np.linspace(0, 1, len(contour.collections))
        # Manually adjust alpha per level
        for i, collection in enumerate(contour.collections):
            alpha_value = alpha_values[i]# (i + 1) / len(contour.collections)  # Increasing alpha for higher levels
            collection.set_alpha(alpha_value)
            # Overlay trajectory
        if labels:
            if (not zoomed) or ((mu[0]>zoomed_xmin) and (mu[0]<zoomed_xmin) and (mu[1]>zoomed_ymin) and (mu[1]<zoomed_ymin)):
                if c!=target_class:
                    ax.text(mu[0], mu[1],class_names[c], color='black', zorder=20,
                          bbox={'facecolor':'white','alpha':0.8,'edgecolor':'none','pad':1},
                          ha='center', va='center', fontsize = 48) 
                else:
                    ax.text(mu[0], mu[1]+0.2,class_names[c], color='black',zorder=20,
                          bbox={'facecolor':'white','alpha':0.8,'edgecolor':'none','pad':1},
                          ha='center', va='center', fontsize = 48) 
    return ax, xlim, ylim


def plot_vae_trajectory(vae, trajectories, target_class, class_names, 
                          figsize=10, zoom_radius = 1, mark_origin=False):
    cmaps = [
    "coolwarm", "viridis", "plasma", "cividis", "magma", 
    "inferno", "Blues", "Greens", "Purples", "Reds"]
    # Create figure and axs[0]is
    
    labels = list(vae.prototypes.keys())
    latents_stdvar = {k: v[1] for k, v in vae.prototypes.items()}
    fig1, axs1 = plt.subplots(figsize=(figsize*2, figsize))
    fig2, axs2 = plt.subplots(figsize=(figsize*2, figsize))
    # traj = np.cumsum(np.random.randn(300, 2) * 0.5, axs[0]is=0)  # Random walk
    axs = [axs1, axs2]
    markersize = 0 if len(trajectories[0])< 100 else 5
    
    axs[0], xlim, ylim = plot_gaussians(vae.prototypes, class_names, target_class,  cmaps, axs[0], zoomed=True, zoom_radio = zoom_radius)
        # Plot contour
    xlim = np.array(xlim)
    ylim = np.array(ylim)
    axs[0].set_xlim(min(xlim[:,0]), max(xlim[:,1]))  # Force visible range
    axs[0].set_ylim(min(ylim[:,0]), max(ylim[:,1]))  # Force visible range  
    
    axs[1], _,_ = plot_gaussians(vae.prototypes, class_names, target_class, cmaps, axs[1])
    

    mu, sigma = vae.prototypes[target_class]
    mu = mu.flatten()
    # mu = mu.numpy().flatten()
    colors = [cplt.to_hex(cm.tab10(i)) for i in range(len(trajectories[0]))]
    for i, traj in enumerate(trajectories):
        axs[1].plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, zorder=11, alpha=0.6)  
        axs[0].scatter(traj[-1, 0], traj[-1, 1], color=colors[i], label="End", s=1200, zorder=16, edgecolor='black', marker = "*")
        
        axs[0].plot(traj[:, 0], traj[:, 1], marker='o', markersize=8,   color=colors[i], linewidth=1, zorder=11, alpha=0.6, label =f'Traj. {i+1}' ) 
        axs[1].scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=1200, zorder=16, edgecolor='black', marker = "*")
        
    axs[1].scatter(mu[0], mu[1], color='black', label="Prototype", s=1200, zorder=15, edgecolor='white', marker = "*")
    if mark_origin: 
        axs[1].plot(0, 0, color='black', zorder=15, ms=20, marker='x', mec='black', mew=5, ls="none")
        # axs[0].scatter(0, 0, color='black', s=500, zorder=15, marker = "X")
    axs[0].scatter(mu[0], mu[1], color='black', label="Prototype", s=1200, zorder=15, edgecolor='white', marker = "*")
    
    axs[0].set_xlim(mu[0] - zoom_radius, mu[0] + zoom_radius)
    axs[0].set_ylim(mu[1] - zoom_radius, mu[1] + zoom_radius)
    # axs[0]s[1].set_xlim(x_min, x_maxs[0])  # Force visible range
    # axs[0]s[1].set_ylim(y_min, y_maxs[0])  # Force visible range  
    legend = axs[0].legend(title=f"Towards {class_names[target_class]}", fontsize =46 )
    legend.get_title().set_fontsize('46') #legend 'Title' fontsize
    return fig1, axs1, fig2, axs2, colors
#%%
from matplotlib.collections import LineCollection      
def plot_vae2d_random_trajectory(vae_2d, trajectories, train_loader, target_class, class_names, ax,
                          figsize=10, zoom_radius = 3, mark_origin=False):
    # Initializing the random seed
    # random_seed=1000
    cmaps = ["inferno_r", "viridis_r", "coolwarm_r", "Blues", "Greens", "Purples", "Reds",
      "plasma", "cividis", "magma", 
     ]
    # Create figure and axs[0]is
    cmaps_gaussians= [
    "coolwarm", "viridis", "plasma", "cividis", "magma", 
    "inferno", "Blues", "Greens", "Purples", "Reds"]
    colors = ['red', 'purple','blue']
    latents_mean, latents_stdvar, labels = get_data_predictions(vae_2d, train_loader)
    classes_mean = get_classes_mean(train_loader, labels, latents_mean, latents_stdvar)
    # fig, ax = plt.subplots(figsize=(figsize, figsize))
    # markersize = 0 if len(trajectory)< 100 else 5
    
    ax, _,_ = plot_gaussians(classes_mean, class_names, target_class,  cmaps_gaussians, ax, labels = False)
    # plt.subplots_adjust(hspace=0)
    for i, trajectory in enumerate(trajectories[:3]):
        points = np.array(trajectory).reshape(-1, 1, 2)
        segments = np.hstack([points[:-1], points[1:]])  # Create line segments
        # Create a colormap gradient (normalize by segment index)
        norm = plt.Normalize(0, len(segments))
        # colors = plt.cm.viridis(norm(np.arange(len(segments))))
        lc = LineCollection(segments, cmap=cmaps[i], norm=norm, linewidth=2)
        print(len(trajectory), len(segments))
        lc.set_array(np.arange(len(segments)))  # Use segment index for color mapping
        ax.add_collection(lc)
        # ax.scatter(traj[-1, 0], traj[-1, 1],  label="End", s=200, zorder=16, edgecolor='black', marker = "^")
        ax.scatter(0,0,  s=300, zorder=16, color='black', marker = "*")
        ax.scatter(trajectory[-1][0],trajectory[-1][1],  s=800, zorder=16, color=colors[i], edgecolor='white', linewidth=2, marker = "*")
        # ax.add_collection(lc)
        
        cbar = plt.colorbar(lc, ax=ax)
        
        cbar.set_ticks([])
        mu, sigma = classes_mean[target_class]
        mu = mu.numpy().flatten()
        # colors = [cplt.to_hex(cm.tab10(i)) for i in range(len(trajectory))]
    # ax.scatter(trajectory[-1], trajectory[-1], color='black', label="End", s=1200, zorder=16, edgecolor='black', marker = "*")
        if i==0: cbar.set_label("Direction", fontsize=28)
    
    ax.scatter(mu[0], mu[1], color='black', label="Prototype", s=1200, zorder=15, edgecolor='white', marker = "*")
    ax.axis('off')
    # if mark_origin: 
    #     axs[1].plot(0, 0, color='black', zorder=15, ms=20, marker='x', mec='black', mew=5, ls="none")
    #     # ax.scatter(0, 0, color='black', s=500, zorder=15, marker = "X")
    
    # ax.set_xlim(mu[0] - zoom_radius, mu[0] + zoom_radius)
    # ax.set_ylim(mu[1] - zoom_radius, mu[1] + zoom_radius)
    # axs[1].set_xlim(x_min, x_max)  # Force visible range
    # axs[1].set_ylim(y_min, y_max)  # Force visible range  ' fontsize
    return ax

#%%
# This combines figures from different calls to minimal_loop (note that figures are rendered as images and can not be further modified)
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

#%%
"""
Legacy code
"""

def plot_vae2d_trajectory_legacy(vae_2d, trajectory, train_loader, target_class, class_names, 
                          figsize=10, zoom_radius = 1, mark_origin=False):
    # Initializing the random seed
    random_seed=1000
    cmaps = [
    "coolwarm", "viridis", "plasma", "cividis", "magma", 
    "inferno", "Blues", "Greens", "Purples", "Reds"]
    # Create figure and axs[0]is
    
    
    latents_mean, latents_stdvar, labels = get_data_predictions(vae_2d, train_loader)
    classes_mean = get_classes_mean(train_loader, labels, latents_mean, latents_stdvar)
    fig1, axs1 = plt.subplots(figsize=(figsize*2, figsize))
    fig2, axs2 = plt.subplots(figsize=(figsize*2, figsize))
    # traj = np.cumsum(np.random.randn(300, 2) * 0.5, axs[0]is=0)  # Random walk
    axs = [axs1, axs2]
    markersize = 0 if len(trajectory)< 100 else 5
    
    axs[0], xlim, ylim = plot_gaussians(classes_mean, class_names, target_class,  cmaps, axs[0], zoomed=True, zoom_radio = zoom_radius)
        # Plot contour
    xlim = np.array(xlim)
    ylim = np.array(ylim)
    axs[0].set_xlim(min(xlim[:,0]), max(xlim[:,1]))  # Force visible range
    axs[0].set_ylim(min(ylim[:,0]), max(ylim[:,1]))  # Force visible range  
    
    axs[1], _,_ = plot_gaussians(classes_mean, class_names, target_class, cmaps, axs[1])
    

    mu, sigma = classes_mean[target_class]
    mu = mu.numpy().flatten()
    colors = [cplt.to_hex(cm.tab10(i)) for i in range(len(trajectory))]
    for i, traj in enumerate(trajectory):
        axs[1].plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=2, zorder=11, alpha=0.6)  
        axs[0].scatter(traj[-1, 0], traj[-1, 1], color=colors[i], label="End", s=1200, zorder=16, edgecolor='black', marker = "*")
        
        axs[0].plot(traj[:, 0], traj[:, 1], marker='o', markersize=8,   color=colors[i], linewidth=1, zorder=11, alpha=0.6, label =f'Traj. {i+1}' ) 
        axs[1].scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=1200, zorder=16, edgecolor='black', marker = "*")
        
    axs[1].scatter(mu[0], mu[1], color='black', label="Prototype", s=1200, zorder=15, edgecolor='white', marker = "*")
    if mark_origin: 
        axs[1].plot(0, 0, color='black', zorder=15, ms=20, marker='x', mec='black', mew=5, ls="none")
        # axs[0].scatter(0, 0, color='black', s=500, zorder=15, marker = "X")
    axs[0].scatter(mu[0], mu[1], color='black', label="Prototype", s=1200, zorder=15, edgecolor='white', marker = "*")
    
    axs[0].set_xlim(mu[0] - zoom_radius, mu[0] + zoom_radius)
    axs[0].set_ylim(mu[1] - zoom_radius, mu[1] + zoom_radius)
    # axs[0]s[1].set_xlim(x_min, x_maxs[0])  # Force visible range
    # axs[0]s[1].set_ylim(y_min, y_maxs[0])  # Force visible range  
    legend = axs[0].legend(title=f"Towards {class_names[target_class]}", fontsize =46 )
    legend.get_title().set_fontsize('46') #legend 'Title' fontsize
    return fig1, axs1, fig2, axs2, colors