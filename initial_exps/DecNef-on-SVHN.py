#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:25:46 2025

@author: alexolza
"""
import torch
from tqdm import tqdm
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from generators import train_model, get_data_predictions, get_classes_mean, show_image
from generators import traverse_two_latent_dimensions, visualize_latent_space
from decnef_loops import minimal_loop
from discriminators import CNNClassification, BinaryDataLoader
import itertools
from torchvision.datasets import ImageFolder
#%%
def check_for_vae_collapse(vae, TGTtrain_loader):
    visualize_latent_space(vae, TGTtrain_loader,
                           device='cuda' if torch.cuda.is_available() else 'cpu',
                           method='UMAP', num_samples=10000)

#%%
device='cuda'
batch_size=64
resolution = 112
TGT_transform = transforms.Compose([
    transforms.Resize((resolution, resolution)),
    transforms.Grayscale(num_output_channels=1),

    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5]),
])
SRC_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.ToTensor(),
    transforms.Resize((resolution, resolution)),
    # transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5]),
    ])
# Download and load the MNIST training data
# TGTtrain_ = datasets.SVHN('.', download=True, split='train', transform=TGT_transform)
# TGTextra_ = datasets.SVHN('.', download=True, split='extra', transform=TGT_transform)
# SVHNtest = datasets.TGT('.', download=True, split='test', transform=TGT_transform)
TGTtrain = ImageFolder(root='./MNIST-M/training', transform = TGT_transform)
# TGTtrain = torch.utils.data.ConcatDataset([TGTtrain_, TGTextra_])
TGTtrain_loader = DataLoader(TGTtrain, batch_size=64, shuffle=True)
# TGTtest_loader = DataLoader(TGTtest, batch_size=64, shuffle=True)

SRCtrain = datasets.MNIST('.', download=True, train=True, transform=SRC_transform)
SRCtrain_loader = DataLoader(SRCtrain, batch_size=64, shuffle=True)

# Download and load the SRC test data
SRCtest = datasets.MNIST('.', download=True, train=False, transform=SRC_transform)
SRCtest_loader = DataLoader(SRCtest, batch_size=64, shuffle=True)


for trl in (TGTtrain_loader, SRCtrain_loader):
    test_img, test_lb = next(iter(trl))
    print(test_lb)
    for i in range(3):
        plt.imshow(test_img[i, 0], cmap='gray')
        plt.show()
#%%
# target_domain = 'sketch'
# source_domain = 'cartoon'
# PACS_source_dir = f'./PACS/{source_domain}'
# PACS_target_dir = f'./PACS/{target_domain}'

# SRC = ImageFolder(root=PACS_source_dir, transform = SRC_transform)
# TGTtrain = ImageFolder(root=PACS_target_dir, transform = TGT_transform)

# # n = len(SRC)  # total number of examples
# # n_test = int(0.1 * n)  # take ~10% for test
# # SRCtest = torch.utils.data.Subset(SRC, range(n_test))  # take first 10%
# # SRCtrain = torch.utils.data.Subset(SRC, range(n_test, n))  # take the rest
# SRCtrain_, SRCtest_ = random_split(SRC, [0.9, 0.1])
# SRCtrain = SRCtrain_.dataset
# SRCtest = SRCtest_.dataset

# SRCtrain_loader = DataLoader(SRCtrain, batch_size=64, shuffle=True)
# SRCtest_loader = DataLoader(SRCtest, batch_size=64, shuffle=True)
# TGTtrain_loader = DataLoader(TGTtrain, batch_size=64, shuffle=True)
# shape = torch.Size([227, 227])
# for trl in (SRCtrain_loader, SRCtest_loader):
#     test_img, test_lb = next(iter(trl))
#     print(test_lb)
#     for i in range(6):
#         plt.imshow(test_img[i, 0], cmap='gray')
#         plt.show()

# assert TGTtrain.class_to_idx == SRCtrain.class_to_idx
# assert False
#%%
z_dim = 128
gen_epochs = 15
vae, vae_history = train_model(TGTtrain_loader,batch_size, epochs=gen_epochs, z_dim=z_dim)
print('GENERATOR TRAINING FINISHED WITH z_dim=',z_dim)
vae_history.BCE_loss.plot()
plt.show()
vae_history.KL_div.plot()
plt.show()
#%%
img, lbl = next(iter(TGTtrain))
plt.imshow(img[0], cmap='gray') , plt.show()
plt.imshow(vae(img.unsqueeze(1).to(device))[0].detach().cpu()[0,0], cmap='gray') , plt.show()
shape = img.shape[-2:]
#%%
visualize_latent_space(vae, TGTtrain_loader,
                       device='cuda' if torch.cuda.is_available() else 'cpu',
                       method='UMAP', num_samples=10000)
#%%
latents_mean, latents_stdvar, labels = get_data_predictions(vae, TGTtrain_loader)
classes_mean = get_classes_mean(TGTtrain_loader, labels, latents_mean, latents_stdvar)

"""
VAE COLLAPSE!!!
THE PROTOTYPES ARE ALWAYS THE SAME
[tensor([[-2.7982e-05, -2.1604e-05,  1.0385e-06,  6.2947e-06,  1.7235e-05,
           3.3745e-05,  2.7007e-05, -1.2681e-07, -7.6378e-06, -4.0379e-06,
           4.4882e-06,  9.4714e-06, -1.5809e-05,  3.6113e-05,  9.7049e-06,
          -2.3841e-05]]),
 tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])]
"""
# try:
#     class_labels = np.array([[k,v] for k,v in TGTtrain.class_to_idx.items()]).T
# except:
class_labels = [i for i in range(10)]

fig, axs = plt.subplots(1, len(class_labels), figsize = (20, 30))
axss = axs.flatten()
i=0

    
for label, idx in enumerate(class_labels):
    latents_mean_target, latents_stddev_target = classes_mean[int(idx)]
    target_prototype = vae.decoder(torch.Tensor(latents_mean_target).to(device), vae.target_size)
    axss[i]=show_image(target_prototype[0].detach().cpu(),axss[i], title=f'{label} - {idx}')
    i+=1
plt.show()   
# print(classes_mean)
#%%
z_dim = 128
gen_epochs = 15
vae, vae_history = train_model(TGTtrain_loader,batch_size, epochs=gen_epochs, z_dim=z_dim)
print('GENERATOR TRAINING FINISHED WITH z_dim=',z_dim)
vae_history.BCE_loss.plot()
plt.show()
vae_history.KL_div.plot()
plt.show()
#%%
img, lbl = next(iter(TGTtrain))
plt.imshow(img[0], cmap='gray') , plt.show()
plt.imshow(vae(img.unsqueeze(1).to(device))[0].detach().cpu()[0,0], cmap='gray') , plt.show()
shape = img.shape[-2:]
#%%
latents_mean, latents_stdvar, labels = get_data_predictions(vae, TGTtrain_loader)
classes_mean = get_classes_mean(TGTtrain_loader, labels, latents_mean, latents_stdvar)

"""
VAE COLLAPSE!!!
THE PROTOTYPES ARE ALWAYS THE SAME
[tensor([[-2.7982e-05, -2.1604e-05,  1.0385e-06,  6.2947e-06,  1.7235e-05,
           3.3745e-05,  2.7007e-05, -1.2681e-07, -7.6378e-06, -4.0379e-06,
           4.4882e-06,  9.4714e-06, -1.5809e-05,  3.6113e-05,  9.7049e-06,
          -2.3841e-05]]),
 tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])]
"""
# try:
#     class_labels = np.array([[k,v] for k,v in TGTtrain.class_to_idx.items()]).T
# except:
class_labels = [i for i in range(10)]

fig, axs = plt.subplots(1, len(class_labels), figsize = (20, 30))
axss = axs.flatten()
i=0

    
for label, idx in enumerate(class_labels):
    latents_mean_target, latents_stddev_target = classes_mean[int(idx)]
    target_prototype = vae.decoder(torch.Tensor(latents_mean_target).to(device), vae.target_size)
    axss[i]=show_image(target_prototype[0].detach().cpu(),axss[i], title=f'{label} - {idx}')
    i+=1
plt.show() 
print({k: v[0][:3] for k,v in classes_mean.items()})  
#%%
visualize_latent_space(vae, TGTtrain_loader,
                       device='cuda' if torch.cuda.is_available() else 'cpu',
                       method='UMAP', num_samples=10000)
#%%
vae_2d, vae_2d_history = train_model(TGTtrain_loader, batch_size, epochs=20, z_dim=2)
print('GENERATOR TRAINING FINISHED WITH z_dim=',2)
vae_2d_history.BCE_loss.plot()
plt.show()
vae_2d_history.KL_div.plot()
plt.show()

#%%
z_dist = Normal(torch.zeros(1, 2), torch.ones(1, 2))
input_sample = torch.zeros(1, 2)

traverse_two_latent_dimensions(vae_2d, input_sample, z_dist, n_samples=20, dim_1=0, dim_2=1, z_dim=2, digit_size = resolution,
                               title=f'traversing 2D latent space', device=device)
#%%
# TODO: FIX THIS!!!! I WILL NEED TO MAP TARGET CLASS TO 1 OR ZERO BECAUSE THE DISCRIMINATOR RELABELS CLASSES. I could reuse/integrate binarydataloader.relabel
"""
 p = torch.nn.Softmax()(discriminator(X0).flatten())[target_class]

IndexError: index 2 is out of bounds for dimension 0 with size 2
"""
class_combinations = list(itertools.combinations(range(10), 2))
for combo in tqdm(class_combinations):
    print('DISCRIMINATOR TRAINING FOR CLASSES ',combo)
    # combo=[0,1]
    # binary_train_loader = binary_loader(SRCtrain, combo, batch_size=16)
    # binary_test_loader = binary_loader(SRCtest, combo, batch_size=16)
    # Train binary discriminator
    tl = BinaryDataLoader(SRCtrain, list(combo), batch_size=16)
    testl = BinaryDataLoader(SRCtest, list(combo), batch_size=16)
    discriminator = CNNClassification(shape, combo,device)  # SRCtrain[0][0].shape
    # discriminator = to_device(discriminator,device)
    discriminator.evaluate(testl)
    discriminator.fit( epochs=10, lr=1e-3, train_loader=tl, val_loader = testl)
    discriminator.plot_accuracies()
    plt.show()
    
    n_iter = 500
    lambda_ = 0.01
    # Once the generator and the discriminator have been trained do the loop:
    for target_class in combo:
        p_vae = minimal_loop(TGTtrain_loader, vae, discriminator, z_dim, target_class, lambda_, n_iter, device)
        p_vae2d = minimal_loop(TGTtrain_loader, vae_2d, discriminator, 2, target_class, lambda_, n_iter, device)
        

