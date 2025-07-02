#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:12:20 2025

@author: alexolza
"""
import sys
sys.path.append('..')
import torch
import os
import regex as re
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
###########################
from components.discriminators import CNNClassification, BinaryDataLoader
from config_files.traditional_decnef_n_instances import traditional_decnef_n_instances_parser
#%%
global_random_seed = 42
config = traditional_decnef_n_instances_parser()

device = 'cuda'
outpath = f'../EXPERIMENTS/{config.EXP_NAME}/output/'
modelpath = f'../EXPERIMENTS/{config.EXP_NAME}/weights/'

discriminator_epochs = 10
discriminator_batch_size = 16
tgt_non_tgt = [config.target_class_idx, config.non_target_class_idx]

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
    print(f'Saved {discriminator_fname}, exiting')
else:
    print(f'Found {discriminator_fname}, exiting')