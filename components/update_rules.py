#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 15:58:15 2025

@author: alexolza
"""
import torch
import numpy as np
from torch.distributions import MultivariateNormal

# def update_z_fixed_normal_drift_adaptive_variance(z, p, lambda_val, f_p, noise_sigma_0 = 10, **f_p_kwargs): #regression towards 0, 0
#     noise_sigma = noise_sigma_0 * f_p(p, reverse=True,**f_p_kwargs)
#     noise_cov = noise_sigma*np.eye(z.size)
#     noise_dist = multivariate_normal([0,0], noise_cov)
#     normal_update = noise_dist.rvs()
#     z_new = (1-lambda_val) *  z + lambda_val * normal_update
#     return z_new, noise_sigma

def identity_f_p(p, reverse):
    return 1-p if reverse else p
def linsig(p, p0, eps=1e-6, reverse = True): 
    p_ = 1-p if reverse else p
    p0_ = 1-p0 if reverse else p0
    return ((p0_)/(p_+eps))

def exp_sig(p,p0, sigma0=1, k=2, eps=1e-6, max_value= 2, reverse=False):
    scale = min(max_value, np.exp(k*(p0-p)/(p+eps)))
    # print(p0, p, scale)
    return scale

#exp 1
# def powsig(p, p0,k=2, eps=1e-3, sigma0=1, reverse = False): 
#     # p_ = 1-p if reverse else p
#     # p0_ = 1-p0 if reverse else p0
#     eps = min(p0,p)
#     # scale = ((p0+eps)/(p+eps))**k
#     scale = ((1-p)**2)*((p0+eps)/(p+eps))**k
#     return scale.item()#min(2*sigma0,scale)

# exp 3   
# def powsig(p, p0,k=2, eps=1e-3, sigma0=1, reverse = False): 
#     # p_ = 1-p if reverse else p
#     # p0_ = 1-p0 if reverse else p0
#     eps = min(p0,p)
#     area0 = p0**2
#     area = p**2
#     scale =((1-p)**2)*((area0+eps)/(area+eps))**k  # If the new p is high, this will be low
#     return scale.item()#min(2*sigma0,scale)

# exp 4  
# def powsig(p, p0,k=2, eps=1e-3, sigma0=1, reverse = False): 
#     # p_ = 1-p if reverse else p
#     # p0_ = 1-p0 if reverse else p0
#     eps = min(p0,p)
#     # area0 = p0**2
#     # area = p**2
#     scale = ((1-p0)**2)*((p0+eps)/(p+eps))**k  # If the old p is high, this will be low
#     return scale.item()#min(2*sigma0,scale)

def powsig(p, p0,k=2, eps=1e-3, sigma0=1, reverse = False): 
    # p_ = 1-p if reverse else p
    # p0_ = 1-p0 if reverse else p0
    eps = torch.min(p0.cpu(),p.cpu())
    # scale = ((p0+eps)/(p+eps))**k
    scale = ((1-p.cpu())**2)*((p0.cpu()+eps)/(p.cpu()+eps))**k
    return scale.item()#min(2*sigma0,scale)

def fp0(p, p0, k=1.05, reverse=True):
     return np.where(p<=p0, k+p*(1-k)/p0, -(p/(1-p0))+(1/(1-p0)))
 
def sigm_p_np(p, slope, reverse=False):  # Controls movement towards the mean
    x = 0.5-p if reverse else p-0.5
    return (1/(1+np.exp(-slope*x)))#  p

def sigm_p(p, slope, reverse=False):  # Controls movement towards the mean
    x = 0.5-p if reverse else p-0.5
    return (1/(1+torch.exp(-slope*x)))#  p

# def update_z_fixed_normal_drift_scipy(z, p, target_dist, lambda_val, f_p, noise_mu=None, noise_sigma =None, **kwargs):
#     noise_cov = np.diag(noise_sigma)
#     noise_dist = multivariate_normal(noise_mu, noise_cov)
#     det_upd_normal =  f_p(p, reverse=False, **kwargs) * target_dist.rvs()
#     normal_update = f_p(p, reverse=True, **kwargs)* noise_dist.rvs()
#     z_new = (1-lambda_val) *  (normal_update +z) + lambda_val * det_upd_normal
#     return z_new

def update_z_fixed_normal_drift(z, p, target_dist, lambda_val, f_p, device='cuda', noise_mu=None, noise_sigma =None, **f_p_kwargs):
    if noise_sigma is None: noise_sigma = torch.ones(z.shape)
    if noise_mu is None: noise_mu = torch.zeros(z.shape)
    assert noise_mu.shape==z.shape, print(noise_mu.shape, z.shape)
    assert noise_sigma.shape==z.shape, print(noise_sigma.shape, z.shape)
    noise_cov = noise_sigma*torch.eye(z.shape[-1])
    noise_dist = MultivariateNormal(noise_mu, covariance_matrix=noise_cov)
    det_upd_normal =  f_p(p, reverse=False, **f_p_kwargs).to(device) * target_dist.sample().to(device)
    normal_update = f_p(p, reverse=True, **f_p_kwargs).to(device)* noise_dist.sample().to(device)
    z_new = (1-lambda_val) *  (normal_update +z.to(device)) + lambda_val * det_upd_normal
    return z_new

def update_z_moving_normal_drift(z, p, target_dist, lambda_val, f_p, device='cuda', noise_sigma =torch.tensor([[1.0,1.0]]), **f_p_kwargs):
    assert noise_sigma.shape==z.shape, print('shape mismatch: ', noise_sigma.shape, z.shape)
    noise_cov = noise_sigma*torch.eye(z.shape[-1]).cpu()
    noise_dist = MultivariateNormal(z.cpu(), covariance_matrix=noise_cov)
    t = target_dist.sample().to(device)
    det_upd_normal =  f_p(p, reverse=False, **f_p_kwargs).to(device) * t
    normal_update = f_p(p, reverse=True, **f_p_kwargs).to(device)* noise_dist.sample().to(device)
    z_new = (1-lambda_val) *  (normal_update +z.to(device)) + lambda_val * det_upd_normal
    return z_new

def update_z_moving_normal_drift_adaptive_variance(trajectory, p, p0, lambda_val, f_p, warm_up=False, device='cuda', max_sigma=1, noise_sigma_0 =1, seed=0, **f_p_kwargs):
    z  = torch.tensor(trajectory[-1])
    noise_sigma = ((1-lambda_val) *noise_sigma_0 + lambda_val * f_p(p.cpu(), p0.cpu(), **f_p_kwargs)).cpu()
    noise_sigma = torch.tensor(min(noise_sigma.item(), max_sigma)).cpu()
    noise_cov = noise_sigma*torch.eye(z.shape[-1]).cpu()
    noise_dist = MultivariateNormal(z.cpu(), covariance_matrix=noise_cov)
    normal_update = noise_dist.sample().to(device)
    z_new =  (1-lambda_val) * z.to(device) + lambda_val * normal_update
    # z_new =  (1-lambda_val) * p * z.to(device) + lambda_val * (1-p) * normal_update # regression to 0,0, WHY?????? seemed to make sense :(
    # z_new =  (1-lambda_val * p) * z.to(device) + lambda_val * (1-p) * normal_update # blowing dist and sigma!!!!!!!!!
    return z_new, noise_sigma

def update_z_moving_normal_drift_adaptive_variance_memory(trajectory, p, p0, lambda_val, f_p, warm_up = False, device='cuda', max_sigma=1, noise_sigma_0 =1, seed=0, verbose=False, **f_p_kwargs):
    z  = torch.tensor(trajectory[-1])
    retreat = True if ((p<0.75*p0) and (not warm_up)) else False
    if retreat: # This will be true if z_{i+1} is worse than z_i
        if verbose: print(f'Retreat p/p0={p/p0}')
        z_previous = torch.tensor(trajectory[-2])    
        # Since z_{i+1} was bad, we return to z_i and we adopt a more exploratory strategy 
        noise_sigma = ((1-lambda_val) *noise_sigma_0 + lambda_val * f_p(p.cpu(), p0.cpu(), **f_p_kwargs)).cpu()
        # Bad z_{i+1} should make noise_sigma > noise_sigma_0, but just in case we take the maximum
        # noise_sigma = torch.tensor(min(max(noise_sigma.item(), noise_sigma_0.item()),
        #                                max_sigma)).cpu()
        noise_sigma = torch.tensor(min( noise_sigma_0.item(),
                                       max_sigma)).cpu()
        noise_cov = noise_sigma*torch.eye(z.shape[-1]).cpu()
        with torch.random.fork_rng():
            torch.manual_seed(seed)  # Local seed
            noise_dist = MultivariateNormal(z_previous.cpu(), covariance_matrix=noise_cov)
            normal_update = noise_dist.sample().to(device)
        z_new =  (1-lambda_val) * z_previous.to(device) + lambda_val * normal_update
    else:
        z_new, noise_sigma = update_z_moving_normal_drift_adaptive_variance(trajectory, p, p0, lambda_val, f_p, warm_up, device, max_sigma, noise_sigma_0, **f_p_kwargs)
    return z_new, noise_sigma