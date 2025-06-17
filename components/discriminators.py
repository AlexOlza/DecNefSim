#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:53:58 2025

@author: alexolza
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import pandas as pd

def get_default_device():
    """ Set Device to GPU or CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    "Move data to the device"
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)

class DeviceDataLoader(DataLoader):
    """ Wrap a dataloader to move data to a device """
    
    def __init__(self, dataset, device, combo = None, batch_size: int=1, shuffle: bool=None, 
                 sampler=None,drop_last: bool=False, *args, **kwargs):
        
        self.dataset = dataset
        if isinstance(combo,list):
            mask = np.logical_or(self.dataset.targets.numpy()==combo[0], self.dataset.targets.numpy()==combo[1])
            combo_indices = np.argwhere(mask).ravel()
            binary_sampler = SubsetRandomSampler(combo_indices)
            # binary_loader = DataLoader(dataset, batch_size=64, sampler= binary_sampler)
            sampler = binary_sampler
            
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=drop_last,
                         *args, **kwargs)
        
        self.device = device
        self.classes = np.array(combo) if isinstance(combo,list) else self.dataset.targets.unique()
        self.minclass = self.classes.min()
        self.maxclass = self.classes.max()
    def __iter__(self):
        """ Yield a batch of data after moving it to device"""
        for (x, y) in super().__iter__():
            y_ = self.__relabel__(y)
            yield to_device((x,y_),self.device)
    def __relabel__(self, y):
        if len(self.classes)==2:
            return ((y-self.minclass)/(self.maxclass-self.minclass)).type(torch.LongTensor)
        else: return y.type(torch.LongTensor)
    def __len__(self):
        """ Number of batches """
        return super().__len__()

    def binarize(self, combo):
        pass
    
class BinaryDataLoader(DataLoader):
    
    def __init__(self, dataset, combo = None, batch_size: int=1, shuffle: bool=None, 
                 sampler=None,drop_last: bool=False, *args, **kwargs):
        
        self.dataset = dataset
        if isinstance(combo,list):
            mask = np.logical_or(self.dataset.targets.numpy()==combo[0], self.dataset.targets.numpy()==combo[1])
            combo_indices = np.argwhere(mask).ravel()
            binary_sampler = SubsetRandomSampler(combo_indices)
            # binary_loader = DataLoader(dataset, batch_size=64, sampler= binary_sampler)
            sampler = binary_sampler
            
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=drop_last,
                         *args, **kwargs)

        self.classes = np.array(combo) if isinstance(combo,list) else self.dataset.targets.unique()
        self.minclass = self.classes.min()
        self.maxclass = self.classes.max()
    def __iter__(self):
        """ Yield a batch of data after moving it to device"""
        for (x, y) in super().__iter__():
            y_ = self.__relabel__(y)
            yield (x,y_)
    def __relabel__(self, y):
        if len(self.classes)==2:
            return ((y-self.minclass)/(self.maxclass-self.minclass)).type(torch.LongTensor)
        else: return y.type(torch.LongTensor)
    def __len__(self):
        """ Number of batches """
        return super().__len__()

    def binarize(self, combo):
        pass
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def __init__(self, device, name='CNN classifier'):
        super().__init__()
        self.device = device
        self.name = name
    def training_step(self, batch):
        images, labels = batch 
        images, labels = images.to(self.device), labels.to(self.device)
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        images, labels = images.to(self.device), labels.to(self.device) 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        if self.verbose: 
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


    def plot_accuracies(self, title = '', ax = None):
        """ Plot the history of accuracies"""
        accuracies = [x['val_acc'] for x in self.history]
        if ax == None: fig, ax = plt.subplots()
        ax.plot(accuracies, '-x')
        ax.set_xlabel('Epoch')
        # plt.ylabel('Accuracy')
        if len(title)>0: ax.set_title(title);
        return ax
      
    @torch.no_grad()
    def evaluate(self, val_loader):
        self.eval()
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)

      
    def fit(self, epochs, lr, train_loader, val_loader, opt_func = torch.optim.SGD, verbose=0):
        self.verbose = verbose
        history = []
        optimizer = opt_func(self.parameters(),lr)
        for epoch in range(epochs):
            
            self.train()
            train_losses = []
            for batch in train_loader:
                loss = self.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            result = self.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            self.epoch_end(epoch, result)
            history.append(result)
        
        self.history = history
    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'extra_attributes': {
                'history': self.history
            }
        }, path)

    def load(self, path, strict=True):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        extras = checkpoint.get('extra_attributes', {})
        for key, value in extras.items():
            setattr(self, key, value)
    def history_to_df(self):
        return pd.DataFrame(self.history,
                            columns = ['train_loss','val_loss','val_acc'])
    
        
class CNNClassification(ImageClassificationBase):
    def __init__(self, image_shape, classes, device='cuda', name = 'CNN classifer'):
        super().__init__(device,name)
        n_channels, resolution_x, resolution_y = image_shape
        self.classes = classes
        n_classes = len(self.classes)
        self.network = nn.Sequential(
            
            nn.Conv2d(n_channels, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            # nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            # nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            # nn.MaxPool2d(2,2),
            nn.Flatten()
            )
        # Dynamically compute the number of input features for the first linear layer
        # Compute dynamically the correct input size for the first fully connected layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, resolution_x, resolution_y)
            dummy_output = self.network(dummy_input)
            linear_input_size = dummy_output.shape[1]  # Get flattened size
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes) # Since F.cross_entropy() expects raw logits, there’s no need for softmax.
        )
        self.to(self.device)
    def forward(self, xb):
        xb = self.network(xb)
        xb = self.fc(xb)
        return xb

def plot_predicted_probability_distribution(discr_dict, img_size, device, niter=1000):
    probs = {c: [] for c in discr_dict.keys()}
    p0 = {}
    for c, discriminator in discr_dict.items():
        p0[c] = torch.nn.Softmax()(discriminator(torch.zeros((1,1,img_size, img_size)).to(device)).flatten())[0].item()
        for i in range(niter):
            X0 = torch.normal(0.,1.,(1,1,img_size, img_size)).to(device)
            p = torch.nn.Softmax()(discriminator(X0).flatten())[0].item()
            probs[c].append(p)
    return pd.DataFrame(probs), p0