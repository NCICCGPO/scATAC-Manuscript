import os
import time
import datetime
import numpy as np
from collections import OrderedDict
import tqdm
import argparse
import pickle

import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

# train
class VAELoss(nn.Module):
    def __init__(self, 
                 beta: float = 1., 
                 reconstruction: str ='cont'):
        """
        Arguments:
          **reconstruction** str (default: `cont`)
            One of 'cont' to trigger mse loss or 'cat' for bce loss
        """
        super().__init__()
        self.beta = beta
        if reconstruction == 'cont':
            self.reconstruction_criterion = nn.MSELoss(reduction='sum')
        elif reconstruction == 'cat':
            self.reconstruction_criterion = nn.BCELoss(reduction='sum')

    
    def forward(self, X, Xhat, mean, logvar):
        reconstruction_loss = self.reconstruction_criterion(Xhat, X)
        KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
        return reconstruction_loss + self.beta*KL_divergence

class EarlyStop:
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0, 
                 save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0
            
        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {"net":model.state_dict(), "optimizer":optimizer.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss