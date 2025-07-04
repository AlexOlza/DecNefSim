#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:41:10 2025

@author: alexolza
"""

import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import random
from torchvision.models import alexnet


def pixel_correlation(img1, img2):
    img1_flat = img1.view(img1.size(0), -1)
    img2_flat = img2.view(img2.size(0), -1)
    corr = torch.nn.functional.cosine_similarity(img1_flat, img2_flat, dim=1)
    return corr.mean().item()

ssim_metric = StructuralSimilarityIndexMeasure()
def compute_ssim(img1, img2):
    return ssim_metric(img1, img2).item()


def get_alexnet_embedding(model, img, layer_name):
    activations = {}
    def hook(module, input, output): activations[layer_name] = output
    handle = dict([
        ("2", model.features[2].register_forward_hook(hook)),
        ("5", model.features[5].register_forward_hook(hook)),
    ])[layer_name]
    _ = model(img)
    handle.remove()
    return activations[layer_name].view(img.size(0), -1)

def two_way_identification(imgs, reconstructions, emb_model = alexnet(pretrained=True).eval(),
                           layer_name = "5", 
                           get_embedding_fn = get_alexnet_embedding):
    correct = 0
    N = imgs.shape[0]
    for i in range(N):
        emb_true = get_embedding_fn(emb_model, imgs[i:i+1], layer_name)
        emb_pos = get_embedding_fn(emb_model, reconstructions[i:i+1], layer_name)
        j = random.choice([x for x in range(N) if x != i])
        emb_neg = get_embedding_fn(emb_model, reconstructions[j:j+1], layer_name)

        sim_pos = F.cosine_similarity(emb_true, emb_pos).item()
        sim_neg = F.cosine_similarity(emb_true, emb_neg).item()
        correct += sim_pos > sim_neg
    return correct / N

def metric_evolution(img1_array, img2, metric_func):
    return [metric_func(torch.Tensor(img1).unsqueeze(0), img2)
               for img1 in img1_array]