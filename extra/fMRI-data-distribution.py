#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 13:18:17 2025

@author: alexolza
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import re
import os
import glob
import random
import bdpy
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns

random.seed(0)
np.random.seed(0)

DATAPATH = '../data'



region = 'FFG'
source_domain = 'perception'
target_domain = 'imagery'
PROBAS = {}
for subject in range(1,19):
    print(f'SUBJECT {subject}')
    fname_PERC = f'{DATAPATH}/{source_domain}/{subject}/{region}.npy'
    fname_IMAG = f'{DATAPATH}/{target_domain}/{subject}/{region}.npy'
    X_PERC = np.load(fname_PERC)
    y_PERC = pd.read_csv(re.sub(f'{region}.npy','events.csv', fname_PERC),
                                        usecols=['trial_idx','target_category', 'run'])
    X_IMAG = np.load(fname_IMAG)
    y_IMAG = pd.read_csv(re.sub(f'{region}.npy','events.csv', fname_IMAG),
                                        usecols=['trial_idx','target_category', 'run'])
    
    rest_PERC = y_PERC.loc[y_PERC.target_category==2]
    task_PERC = y_PERC.loc[y_PERC.target_category!=2]
    X_task_PERC = X_PERC[task_PERC.index]
    task_PERC.index = range(len(task_PERC))
    
    c0_IMAG = y_IMAG.loc[y_IMAG.target_category==0]
    c1_IMAG = y_IMAG.loc[y_IMAG.target_category==1]
    
    c0_PERC = y_PERC.loc[y_PERC.target_category==0]
    c1_PERC = y_PERC.loc[y_PERC.target_category==1]
    
    
    sgkf = StratifiedGroupKFold(n_splits=2)
    
    for i, (train_index, test_index) in enumerate(sgkf.split(X_task_PERC, task_PERC.target_category.values, (task_PERC.trial_idx.astype(str) + task_PERC.run).values)):
        lr = LogisticRegression().fit(X_task_PERC[train_index,:], task_PERC.target_category[train_index])
        ytest = task_PERC.target_category[test_index]
        c0_index = ytest.loc[ytest==0].index
        c1_index = ytest.loc[ytest==1].index
        probas = {'PERC0': lr.predict_proba(X_task_PERC[c0_index,:])[:,0],
                  'PERC1':  lr.predict_proba(X_task_PERC[c1_index,:])[:,0],
                  
                  'IMAG0': lr.predict_proba(X_IMAG[c0_IMAG.index,:])[:,0],
                  'IMAG1':  lr.predict_proba(X_IMAG[c1_IMAG.index,:])[:,0],
                  'REST': lr.predict_proba(X_PERC[rest_PERC.index,:])[:,0]
                  # 'IMAG-REST': lr.predict_proba(X_IMAG[rest_IMAG.index,:])[:,0]
                  }
        # lr.score(X_task_PERC[test_index,:], ytest)
        PROBAS[subject] = probas
        break
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1,2, figsize = (12,4),sharey=True)
    sns.violinplot(probas, cut=0, ax = axs[0])
    axs[0].axhline(y=0.5,ls='--')
    axs[0].set_title(f'Trained on PERC. Acc.: {lr.score(X_task_PERC[test_index,:], ytest):.2f} PERC; {lr.score(X_IMAG, y_IMAG.target_category):.2f} IMAG')
    # print('Acc. Source test: ', )
    # print('Acc. Source train: ', lr.score(X_task_PERC[train_index,:], task_PERC.target_category[train_index]))
    # print('Acc. Target: ', )
    
    sgkf = StratifiedGroupKFold(n_splits=2)
    # PROBAS = {}
    for i, (train_index, test_index) in enumerate(sgkf.split(X_IMAG, y_IMAG.target_category.values, (y_IMAG.trial_idx.astype(str) + y_IMAG.run).values)):
        lr = LogisticRegression().fit(X_IMAG[train_index,:], y_IMAG.target_category[train_index])
        ytest = y_IMAG.target_category[test_index]
        c0_index = ytest.loc[ytest==0].index
        c1_index = ytest.loc[ytest==1].index
        probas = {'IMAG0': lr.predict_proba(X_IMAG[c0_index,:])[:,0],
                'IMAG1':  lr.predict_proba(X_IMAG[c1_index,:])[:,0],
                'PERC0': lr.predict_proba(X_PERC[c0_PERC.index,:])[:,0],
                  'PERC1':  lr.predict_proba(X_PERC[c1_PERC.index,:])[:,0],
                  'REST': lr.predict_proba(X_PERC[rest_PERC.index,:])[:,0],
                  
                  # 'IMAG-REST': lr.predict_proba(X_IMAG[rest_IMAG.index,:])[:,0]
                  }
        # lr.score(X_PERC[test_index,:], ytest)
        # PROBAS[i] = probas
        break
    
    sns.violinplot(probas, cut=0, ax = axs[1])
    axs[1].axhline(y=0.5,ls='--')
    axs[1].set_title(f'Trained on IMAG. Acc.: {lr.score(X_IMAG[test_index,:], ytest):.2f} IMAG; {lr.score(X_task_PERC, task_PERC.target_category):.2f} PERC')
    fig.suptitle(f'Subject {subject}', y=1)
    plt.show()
    # print('Acc. IMAG test: ', )
    # print('Acc. IMAG train: ', lr.score(X_IMAG[train_index,:], y_IMAG.target_category[train_index]))
    # print('Acc. PERC: ', lr.score(X_task_PERC, task_PERC.target_category))
    print('---------------------------------')
    
#%%
mean_probs={}
std_probs = {}
for k in probas.keys():
    if 'PERC' in k: continue
    mean_probs[k] = [np.mean(p[k]) for s, p in PROBAS.items()]
    std_probs[k] = [np.var(p[k]) for s, p in PROBAS.items()]
#%%
df1 = pd.DataFrame(mean_probs)
df2 = pd.DataFrame(std_probs)
# Plotting
fig, ax = plt.subplots(figsize=(8, 5))


ax.errorbar(range(len(df1)), df1['IMAG0'], yerr=df2['IMAG0'], fmt='o-', capsize=5, label=f'IMAG0', alpha=0.3, color='black')
ax.errorbar(range(len(df1)), df1['REST'], yerr=df2['REST'], fmt='o-', capsize=5, label=f'REST',color='red', alpha=1)

# Labels and legend
# ax.xlabel('Index')
ax.set_xticks(range(0,18))
ax.set_xticklabels(range(1,19))
ax.axhline(y=0.5, color='black', ls='--')
ax.set_xlabel('Subject')
plt.legend()
fig.tight_layout()
fig.savefig('fMRI_probability-distribution.pdf')
plt.show()