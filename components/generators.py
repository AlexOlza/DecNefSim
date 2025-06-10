#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:56:30 2025

@author: alexolza

inspired by: 
    https://hackernoon.com/how-to-sample-from-latent-space-with-variational-autoencoder
    https://github.com/qbxlvnf11/conditional-GAN/blob/main/conditional-GAN-generating-fashion-mnist.ipynb
    https://blog.fastforwardlabs.com/2016/08/12/introducing-variational-autoencoders-in-prose-and-code.html
    
"""
from torch import nn
import torch
from tqdm import tqdm
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import numpy as np
import pandas as pd
def kl_divergence_loss(z_dist):
    return kl_divergence(z_dist,
                         Normal(torch.zeros_like(z_dist.mean),
                                torch.ones_like(z_dist.stddev))
                         ).sum(-1).sum()
reconstruction_loss = nn.BCELoss(reduction='sum')



class Encoder(nn.Module):
    def __init__(self, im_chan=1, output_chan=32, hidden_dim=16):
        super(Encoder, self).__init__()
        self.z_dim = output_chan

        self.encoder = nn.Sequential(
            self.init_conv_block(im_chan, hidden_dim),
            self.init_conv_block(hidden_dim, hidden_dim * 2),
            # double output_chan for mean and std with [output_chan] size
            
            self.init_conv_block(hidden_dim * 2, output_chan * 2, final_layer=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def init_conv_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, final_layer=False):
        layers = [
            nn.Conv2d(input_channels, output_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          stride=stride)
        ]
        if not final_layer:
            layers += [
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

    def forward(self, image):
        encoder_pred = self.encoder(image)
        encoding = encoder_pred.view(len(encoder_pred), -1)
        mean = encoding[:, :self.z_dim]
        logvar = encoding[:, self.z_dim:]
        # encoding output representing standard deviation is interpreted as
        # the logarithm of the variance associated with the normal distribution
        # take the exponent to convert it to standard deviation
        return mean, torch.exp(logvar*0.5)
    

class Decoder(nn.Module):
    def __init__(self, z_dim=32, im_chan=1, hidden_dim=64):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.decoder = nn.Sequential(
            self.init_conv_block(z_dim, hidden_dim * 4),
            self.init_conv_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.init_conv_block(hidden_dim * 2, hidden_dim),
            # nn.AdaptiveAvgPool2d((1, 1)),
            self.init_conv_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def init_conv_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
        layers = [
            nn.ConvTranspose2d(input_channels, output_channels,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding)
        ]
        if not final_layer:
            layers += [
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            ]
        else:
            layers += [nn.Sigmoid()]
        return nn.Sequential(*layers)

    def forward(self, z, target_size):
        # Ensure the input latent vector z is correctly reshaped for the decoder
        x = z.view(-1, self.z_dim, 1, 1)
        # Pass the reshaped input through the decoder network
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class VAE(nn.Module):
  def __init__(self, z_dim=32, im_chan=1):
    super(VAE, self).__init__()
    self.z_dim = z_dim
    self.encoder = Encoder(im_chan, z_dim)
    self.decoder = Decoder(z_dim, im_chan)

  def forward(self, images):
    target_size = images.shape[-2:]  # (height, width)
    if not hasattr(self, 'target_size'): self.target_size = target_size
    z_dist = Normal(*self.encoder(images))
    # sample from distribution with reparametarazation trick
    z = z_dist.rsample()
    decoding = self.decoder(z, target_size)
    return decoding, z_dist



def train_model(train_loader, batch_size, epochs=10, z_dim = 16, device='cuda', annealing_epochs = 0, max_beta = 1):
  BCE, KL = [], []
  model = VAE(z_dim=z_dim).to(device)
  model_opt = torch.optim.Adam(model.parameters())
  if annealing_epochs>0: betas = np.linspace(0.1,0.5,annealing_epochs)
  else: betas = max_beta*np.ones(epochs)
  for epoch in range(epochs):
      epoch_bce, epoch_kl = [], []
      beta = betas[epoch] if epoch< len(betas) else max_beta
      print(f"Epoch {epoch} (beta = {beta})")
      for images, step in tqdm(train_loader):
          images = images.to(device)
          model_opt.zero_grad()
          recon_images, encoding = model(images)
          # print(images.shape, recon_images.shape)
          bce = reconstruction_loss(recon_images, images)#/(torch.Tensor([np.prod(list(model.target_size))]).to(device))
          kl = kl_divergence_loss(encoding)
          # print(list(model.target_size), torch.Tensor(np.prod(np.array(list(model.target_size)))))
          loss = bce+ beta*kl # TODO: SET THIS IN TERMS OF TARGET SIZE!!
          loss.backward()
          model_opt.step()
          epoch_bce.append(bce.item())
          epoch_kl.append(kl.item())
      BCE.append(np.mean(epoch_bce))
      KL.append(np.mean(epoch_kl))
      # show_images_grid(images.cpu(), batch_size, title='Input images')
      # show_images_grid(recon_images.cpu(), batch_size, title='Reconstructed images')
  history = pd.DataFrame(np.transpose([BCE, KL]), columns=['BCE_loss', 'KL_div'])
  return model, history


def get_data_predictions(model, data_loader, device='cuda'):
    model.eval()
    latents_mean = []
    latents_std = []
    labels = []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
          mu, std = model.encoder(data.to(device))
          latents_mean.append(mu.cpu())
          latents_std.append(std.cpu())
          labels.append(label.cpu())
    latents_mean = torch.cat(latents_mean, dim=0)
    latents_std = torch.cat(latents_std, dim=0)
    labels = torch.cat(labels, dim=0)
    return latents_mean, latents_std, labels

def get_classes_mean(train_loader, labels, latents_mean, latents_std,):
  classes_mean = {}
  for class_name in np.unique(labels):
    # class_id = train_loader.dataset.class_to_idx[class_name]
    labels_class = labels[labels==class_name]
    latents_mean_class = latents_mean[labels==class_name]
    latents_mean_class = latents_mean_class.mean(dim=0, keepdims=True)

    latents_std_class = latents_std[labels==class_name]
    latents_std_class = latents_std_class.mean(dim=0, keepdims=True)

    classes_mean[class_name] = [latents_mean_class, latents_std_class]
  return classes_mean

  



def dist_classmean_to_random(trainset, classes, class_names, device, niter=1000):
    dist = {c: [] for c in class_names}
    subsets = {}
    sum_means = {}
    for c, n in zip(classes, class_names):
        indices = (trainset.targets.clone().detach()[..., None] == c).any(-1).nonzero(as_tuple=True)[0]
        subsets[n] = torch.utils.data.Subset(trainset, indices)
    
    for name, subset in subsets.items():
        i=0
        for image, _ in subset:
            if i==0:
                sum_means[name] = image.to(device)
            else:
                sum_means[name] =  image.to(device) + sum_means[name]
            i+=1
        sum_means[name] = sum_means[name] / len(subset)
        # print(sum_means[c].shape)
    zeros = torch.zeros(sum_means[class_names[0]].shape).to(device)
    dist0 = {c: [(zeros - sum_means[c]).pow(2).sum().sqrt().item()] for c in class_names}
    for i in range(niter):
        for c in class_names: 
            X0 = torch.normal(0.,1.,sum_means[c].shape).to(device)
            dist[c].append((X0 - sum_means[c]).pow(2).sum().sqrt().item())
    dist = pd.DataFrame(dist)    
    return dist, pd.DataFrame(dist0)

def kl_divergences_to_std_gaussian(trainset, classes, class_names, device, niter=1000):
    dist = {c: [] for c in class_names}
    subsets = {}
    sum_means = {}
    all_imgs = {}
    for c, n in zip(classes, class_names):
        indices = (trainset.targets.clone().detach()[..., None] == c).any(-1).nonzero(as_tuple=True)[0]
        subsets[n] = torch.utils.data.Subset(trainset, indices)
    
    for name, subset in subsets.items():
        i=0
        for image, _ in subset:
            
            if i==0:
                sum_means[name] = image.to(device)
                all_imgs[name] =image
            else:
                sum_means[name] =  image.to(device) + sum_means[name]
                all_imgs[name] = torch.cat((all_imgs[name],image))
            i+=1
        sum_means[name] = sum_means[name] / len(subset)
        # print(sum_means[c].shape)
    zeros = torch.zeros(sum_means[class_names[0]].shape).to(device)
    dist0 = {c: [(zeros - sum_means[c]).pow(2).sum().sqrt().item()] for c in class_names}
    for i in range(niter):
        for c in class_names: 
            X0 = torch.normal(0.,1.,sum_means[c].shape).to(device)
            dist[c].append((X0 - sum_means[c]).pow(2).sum().sqrt().item())
    dist = pd.DataFrame(dist)    
    return dist, pd.DataFrame(dist0)


def compute_kl_multivariate(trainset, classes, class_names, device='cuda'):
    """
    Computes KL divergence between each class distribution in dataset and a standard normal N(0, I).
    """
    kl_divs = {}
    subsets = {}
    sum_means = {}
    all_imgs = {}
    for c, n in zip(classes, class_names):
        indices = (trainset.targets.clone().detach()[..., None] == c).any(-1).nonzero(as_tuple=True)[0]
        subsets[n] = torch.utils.data.Subset(trainset, indices)
    
    for name, subset in subsets.items():
        i=0
        for image, _ in subset:
            
            if i==0:
                sum_means[name] = image.to(device)
                all_imgs[name] =image
            else:
                sum_means[name] =  image.to(device) + sum_means[name]
                all_imgs[name] = torch.cat((all_imgs[name],image))
            i+=1
        sum_means[name] = sum_means[name] / len(subset)

        # mean_c = class_data.mean(dim=0)  # Mean vector (n^2,)
        mean_c = sum_means[name].view( -1)
        num_samples, n, n = all_imgs[name].shape
        class_data = all_imgs[name].view(num_samples, -1)  # Flatten to (num_samples, n^2)
        cov_c = torch.cov(class_data.T)  # Covariance matrix (n^2, n^2)

        # Stabilize covariance matrix
        jitter = 1e-2 * torch.eye(cov_c.shape[0])
        cov_c += jitter.to(cov_c.device)

        # Log determinant (Cholesky more stable)
        cholesky_c = torch.linalg.cholesky(cov_c)
        log_det_term = 2 * torch.sum(torch.log(torch.diag(cholesky_c)))

        trace_term = torch.trace(cov_c)
        mean_term = torch.dot(mean_c, mean_c)

        kl_div = 0.5 * (trace_term - n**2 + mean_term - log_det_term)
        kl_divs[name] = kl_div.item()

    return kl_divs
