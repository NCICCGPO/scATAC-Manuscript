

import torch
import os
import sys
import time
import datetime
import pickle
import numpy as np
import pandas as pd
import glob
from scipy import io, sparse
import tqdm

import torch
import scanpy as sc


class scatac(torch.utils.data.Dataset):
    def __init__(self, adata, split='train', label='batch', return_CNV=False):
        '''
        Arguments:
          adata (sc.AnnData): to reduce memory load, feed pre-loaded adata that is filtered by split, e.g.
            feed adata[adata.obs['split']=='train', :]. REQUIREMENT: CNV must be encoded in adata.var['copy_nb_batch{}'.format(batch)]
          label (str): key in adata.obs slot that is 1d encoding for class label
          
        Returns:
          X, y, X_div_cnv
        '''
        super().__init__()
        self.split = split
        self.label = label
        if split != 'all':
            adata = adata[adata.obs['split']==split, :]
        self.X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(adata.obs[label], dtype=torch.int64)
        CNV = adata.var.loc[:, ('copy_nb_batch' + adata.obs['batch'].astype(str)).to_list()].to_numpy(dtype=np.float32).transpose() # ~30s
        CNV = torch.tensor(CNV, dtype=torch.float32)
        if return_CNV:
            self.CNV = CNV
        else:
            self.X_div_cnv = self.X / (CNV + 1.)
        self.idx = adata.obs.index.to_list()
        self.n_feat = self.X.shape[1]
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.X_div_cnv[idx], self.idx[idx]
    
    
class tcga(torch.utils.data.Dataset):
    def __init__(self, adata, split='train', label='batch', return_CNV=False):
        '''
        Arguments:
          adata (sc.AnnData): to reduce memory load, feed pre-loaded adata that is filtered by split, e.g.
            feed adata[adata.obs['split']=='train', :]. REQUIREMENT: CNV must be encoded in adata.var['copy_nb_batch{}'.format(batch)]
          label (str): key in adata.obs slot that is 1d encoding for class label
          
        Returns:
          X, y, X_div_cnv
        '''
        super().__init__()
        self.split = split
        self.label = label
        if split != 'all':
            adata = adata[adata.obs['split']==split, :]
        self.X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(adata.obs[label], dtype=torch.int64)
        if split == 'train' or split == 'val':
            CNV = adata.var.loc[:, ('copy_nb_batch' + adata.obs['batch'].astype(str)).to_list()].to_numpy(dtype=np.float32).transpose() # ~30s
            CNV = torch.tensor(CNV, dtype=torch.float32)
            if return_CNV:
                self.CNV = CNV
            else:
                self.X_div_cnv = self.X / (CNV + 1.)
        self.idx = adata.obs.index.to_list()
        self.n_feat = self.X.shape[1]
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if self.split == 'test' or self.split == 'all':
            return self.X[idx], self.y[idx], self.idx[idx]
        else:
            return self.X[idx], self.y[idx], self.X_div_cnv[idx], self.idx[idx]
