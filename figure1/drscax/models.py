# preliminaries

import os
import time
import tqdm 
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

import scanpy as sc
import scvi

import torch
import torch.nn as nn
import torch.nn.functional as F

# cAEv01

class FC(nn.Module):
    def __init__(self, 
                 I: int = 50000, 
                 O: int = 256, 
                 D: float = 0.1, 
                 A: nn.Module = nn.LeakyReLU(),
                 norm: nn.Module = nn.LayerNorm(256, elementwise_affine=False),
                 name: str = '1'):
        '''
        Helper class to build FC layers for a simple NN.
        
        Arguments:
          **I** : int (default: `50000`)
            nb input nodes
          **O** : int (default: `256`)
            nb output nodes, typically np.sqrt(I)
          **D** : float (default: `0.1`)
            probability of dropout
          **norm** : nn.Module (default: `nn.LayerNorm(O, elementwise_affine=False`)
            normalization to apply after linear projection. Can be nn.Identity or 
              nn.BatchNorm1d(O, momentum=0.01, eps=0.001). NOTE: the size must match
              output dim in the first argument to Batch/LayerNorm.
          **A** : nn.Module (default: `nn.LeakyReLU`)
            activation function, including output for last layer, 
              e.g., nn.Identity or nn.sigmoid or nn.softmax(*args)
          **name** : str (default: `1`)
            give a name to the layer so that it is stored in the model
          
        '''
        super().__init__()
        
        q = []
        q.append(("Linear_%s" % name, nn.Linear(I, O)))
        if norm is not None:
            q.append(("norm_%s" % name, norm))
        q.append(("act_%s" % name, A))
        q.append(("dropout_%s" % name, nn.Dropout(p=D)))
        
        self.fc = nn.Sequential(OrderedDict(q))
        
    def forward(self, x):
        return self.fc(x)
        
class Encoder(nn.Module):
    def __init__(self, 
                 feat_in: int = 50000, 
                 n_hidden: int = 256,
                 feat_out: int = 64,
                 nb_layers: int = 4,
                 c1_embed_dim: int = 0,
                 inject_c1_eachlayer: bool = False,
                ):
        super().__init__()
        """
        
        Arguments:
          **embed_dim** : int (default: `0`)
            specify the conditional embedding dimension. This can be concatenated and then passed to a mean, logvar
              encoder specification. NOTE: currently ignored
        """
        self.nb_layers = nb_layers
        self.inject_c1_eachlayer = inject_c1_eachlayer
        assert feat_out % 2 == 0, 'Must specify multiple of two due to linear concat at last layer'
        if inject_c1_eachlayer:
            feat_in = feat_out + c1_embed_dim
            n_hidden = n_hidden + c1_embed_dim
        self.fc1 = FC(I=feat_in, O=n_hidden, 
                      norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                      name='e1')
        
        if nb_layers > 2:
            self.fc2 = FC(
                I=n_hidden, O=n_hidden, 
                norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                name='e2')
            self.fc3 = FC(
                I=n_hidden, O=n_hidden, 
                norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                name='e3')
        if nb_layers > 4:
            self.fc4 = FC(
                I=n_hidden, O=n_hidden, 
                norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                name='e4')
            self.fc5 = FC(
                I=n_hidden, O=n_hidden, 
                norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                name='e5')
            self.fc6 = FC(
                I=n_hidden, O=n_hidden, 
                norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                name='e6')
            self.fc7 = FC(
                I=n_hidden, O=n_hidden, 
                norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                name='e7')
        if nb_layers > 8:
            self.fc8 = FC(
                I=n_hidden, O=n_hidden, 
                norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                name='e8')
            self.fc9 = FC(
                I=n_hidden, O=n_hidden, 
                norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                name='e9')
            self.fc10 = FC(
                I=n_hidden, O=n_hidden, 
                norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                name='e10')
            self.fc11 = FC(
                I=n_hidden, O=n_hidden, 
                norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                name='e11')
        self.fc_last_a = FC(I=n_hidden, O=feat_out // 2, 
                      norm=nn.LayerNorm(feat_out // 2, elementwise_affine=False), 
                      A=nn.Identity(), 
                      name='ea')
        self.fc_last_b = FC(I=n_hidden, O=feat_out // 2, 
                      norm=nn.LayerNorm(feat_out // 2, elementwise_affine=False), 
                      A=nn.Identity(), 
                      name='eb')
    
    def forward(self, x, c1=None):
        if self.inject_c1_eachlayer:
            x = torch.cat((x, c1), dim=-1)
        x = self.fc1(x)
        if self.nb_layers > 2:
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
            x = self.fc2(x)
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
            x = self.fc3(x)
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
        if self.nb_layers > 4:
            x = self.fc4(x)
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
            x = self.fc5(x)
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
            x = self.fc6(x)
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
            x = self.fc7(x)
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
        if self.nb_layers > 8:
            x = self.fc8(x)
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
            x = self.fc9(x)
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
            x = self.fc10(x)
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
            x = self.fc11(x)
            if self.inject_c1_eachlayer:
                x = torch.cat((x, c1), dim=-1)
        x = torch.cat((self.fc_last_a(x), self.fc_last_b(x)), dim=-1)
        if c1 is not None:
            return torch.cat((x, c1), dim=-1) # embed this?
        else:
            return x
        
class Decoder_v05(nn.Module):
    def __init__(self, 
                 layer_io: list = [512, 1024, 2048, 4096, 50000],
                 c1_embed_dim: int = 0, # potentially n_classes? See: https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py
                 # inject_c1_eachlayer: bool = False, # not implemented
                 layer_norm: bool = True,
                 sigmoid_out=False):
        super().__init__()
        self.layer_io = layer_io
        layers = []
        for i, l in enumerate(layer_io):
            if i + 1 < len(layer_io):
                if i == 0:
                    feat_in = layer_io[i] + 2*c1_embed_dim # since the last encoder layer is repeated, could do this and concat at end?
                else:
                    feat_in = layer_io[i]
                feat_out = layer_io[i+1]
                layers.append(("FC{}".format(i+1), 
                              FC(I=feat_in,
                                 O=feat_out,
                                 D=0.,
                                 norm=nn.LayerNorm(feat_out, elementwise_affine=False) if layer_norm else None,
                                 A=nn.LeakyReLU(),
                                 name='d{}'.format(i+1))))
                if i + 2 == len(layer_io):
                    # last layer
                    layers.append(("FClast", 
                              FC(I=feat_in,
                                 O=feat_out,
                                 D=0.,
                                 norm=nn.LayerNorm(feat_out, elementwise_affine=False) if layer_norm else None,
                                 A=nn.Sigmoid() if sigmoid_out else nn.Identity(),
                                 name='dlast')))
                    break
        self.decoder = nn.ModuleDict(OrderedDict(layers))
        
    def forward(self, x, c1=None):
        if c1 is not None:
            x = torch.cat((x, c1), dim=-1)
        for l in range(len(self.layer_io) - 2):
            x = self.decoder['FC{}'.format(l+1)](x)
        return self.decoder['FClast'](x)
    
    
    
class Encoder_v05(nn.Module):
    def __init__(self, 
                 layer_io: list = [50000, 4096, 2048, 1024, 512],
                 c1_embed_dim: int = 0,
                 layer_norm: bool = True, 
                 inject_c1_eachlayer: bool = False,
                ):
        super().__init__()
        """
        
        Arguments:
          **embed_dim** : int (default: `0`)
            specify the conditional embedding dimension. This can be concatenated and then passed to a mean, logvar
              encoder specification. NOTE: currently ignored
              
          **norm**: bool (default: `True`)
            whether to apply nn.Identity() or nn.LayerNorm
        """
        self.layer_io = layer_io
        self.inject_c1_eachlayer = inject_c1_eachlayer
        layers = [] 
        assert layer_io[-1] % 2 == 0, 'Need the last layer to be a concatenatation of two'
        for i, l in enumerate(layer_io):
            if i + 1 < len(layer_io):
                layers.append(('FC{}'.format(i+1),
                    FC(I=layer_io[i] + c1_embed_dim if inject_c1_eachlayer or i==0 else layer_io[i], 
                       O=layer_io[i+1],
                       D=0.1,
                       A=nn.LeakyReLU(),
                       norm=nn.LayerNorm(layer_io[i+1], elementwise_affine=False) if layer_norm else None,
                       name='e{}'.format(i+1))))
                if i + 2 == len(layer_io):
                    # last layer 
                    layers.append(('FCa',
                        FC(I=layer_io[i] + c1_embed_dim, 
                           O=layer_io[i+1] // 2,
                           D=0.1,
                           A=nn.Identity(),
                           norm=None,
                           name='ea'.format(i+1))))
                    layers.append(('FCb',
                        FC(I=layer_io[i] + c1_embed_dim, 
                           O=layer_io[i+1] // 2,
                           D=0.1,
                           A=nn.Identity(),
                           norm=None,
                           name='eb'.format(i+1))))
                    break
        self.encoder = nn.ModuleDict(OrderedDict(layers))

    def forward(self, x, c1=None):
        for l in range(len(self.layer_io) - 2):
            if ((self.inject_c1_eachlayer) or (l==0)) and (c1 is not None): # first layer, with condition
                x = torch.cat((x, c1), dim=-1)
            x = self.encoder['FC{}'.format(l+1)](x)
        if self.inject_c1_eachlayer or c1 is not None:
            x = torch.cat((x, c1), dim=-1)
        x = torch.cat((self.encoder['FCa'](x), 
                       self.encoder['FCb'](x)), dim=-1)
        if c1 is not None:
            return torch.cat((x, c1), dim=-1) # embed this?
        else:
            return x
        
class Decoder(nn.Module):
    def __init__(self, 
                 feat_in: int = 64,
                 n_hidden: int = 256,
                 feat_out: int = 50000,
                 nb_layers: int = 4,
                 c1_embed_dim: int = 0,
                 sigmoid_out=False):
        super().__init__()
        self.nb_layers = nb_layers
        self.fc1 = FC(I=feat_in + c1_embed_dim, O=n_hidden, 
                      norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                      D=0., name='d1')
        if nb_layers > 2:
            self.fc2 = FC(I=n_hidden, O=n_hidden, 
                          norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                          D=0., name='d2')
            self.fc3 = FC(I=n_hidden, O=n_hidden, 
                          norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                          D=0., name='d3')
        if nb_layers > 4:
            self.fc4 = FC(I=n_hidden, O=n_hidden, 
                          norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                          D=0., name='d4')
            self.fc5 = FC(I=n_hidden, O=n_hidden, 
                          norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                          D=0., name='d5')
            self.fc6 = FC(I=n_hidden, O=n_hidden, 
                          norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                          D=0., name='d6')
            self.fc7 = FC(I=n_hidden, O=n_hidden, 
                          norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                          D=0., name='d7')
        if nb_layers > 8:
            self.fc8 = FC(I=n_hidden, O=n_hidden, 
                          norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                          D=0., name='d8')
            self.fc9 = FC(I=n_hidden, O=n_hidden, 
                          norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                          D=0., name='d9')
            self.fc10 = FC(I=n_hidden, O=n_hidden, 
                          norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                          D=0., name='d10')
            self.fc11 = FC(I=n_hidden, O=n_hidden, 
                          norm=nn.LayerNorm(n_hidden, elementwise_affine=False),
                          D=0., name='d11')
        self.fc_last = FC(I=n_hidden, O=feat_out, 
                      norm=nn.LayerNorm(feat_out, elementwise_affine=False),
                      A=nn.Sigmoid() if sigmoid_out else nn.Identity(), 
                      D=0.,
                      name='dlast') # output input value matrix rather than prob
        
    def forward(self, x, c1=None):
        if c1 is not None:
            x = torch.cat((x, c1), dim=-1)
        x = self.fc1(x)
        if self.nb_layers > 2:
            x = self.fc2(x)
            x = self.fc3(x)
        if self.nb_layers > 4:
            x = self.fc4(x)
            x = self.fc5(x)
            x = self.fc6(x)
            x = self.fc7(x)
        if self.nb_layers > 8:
            x = self.fc8(x)
            x = self.fc9(x)
            x = self.fc10(x)
            x = self.fc11(x)
        return self.fc_last(x)
    
# cAEv01
class cAE(nn.Module):
    def __init__(self, 
                 feat_in: int = 50000,
                 n_hidden: int = 256,
                 enc_feat_out: int = 64,
                 nb_layers: int = 4,
                 n_c1_class: int = 58,
                 c1_embed_dim: int = 8,
                 inject_c1_eachlayer: bool = False,
                 sigmoid_out: bool = False,
                 return_latent: bool = False):
        super().__init__()
        self.return_latent = return_latent
        
        if c1_embed_dim != 0 and n_c1_class != 0:
            self.c1_embedding = nn.Embedding(n_c1_class, c1_embed_dim)
            
        self.encoder = Encoder(
            feat_in=feat_in, 
            n_hidden=n_hidden,
            feat_out=enc_feat_out,
            nb_layers=nb_layers,
            inject_c1_eachlayer=inject_c1_eachlayer,
            # embed_dim=embed_dim,
        )
        self.decoder = Decoder(
            feat_in=enc_feat_out,
            n_hidden=n_hidden,
            feat_out=feat_in,
            nb_layers=nb_layers,
            c1_embed_dim=c1_embed_dim,
            sigmoid_out=sigmoid_out,
        )
        
    def forward(self, x, c1=None):
        z = self.encoder(x)
        if c1 is not None:
            c1 = self.c1_embedding(c1)
            xhat = self.decoder(torch.cat((z, c1), dim=-1))
        else:
            xhat = self.decoder(z)
        if self.return_latent:
            return xhat, z
        else:
            return xhat
        
        
# cAEv05
class cAE_v05(nn.Module):
    def __init__(self, 
                 layer_io: list = [50000, 4096, 2048, 1024, 512], 
                 layer_norm: bool = True,
                 n_c1_class: int = 58,
                 c1_embed_dim: int = 8,
                 inject_c1_eachlayer: bool = False,
                 sigmoid_out: bool = False,
                 return_latent: bool = False):
        super().__init__()
        self.return_latent = return_latent
        
        if c1_embed_dim != 0 and n_c1_class != 0:
            self.c1_embedding = nn.Embedding(n_c1_class, c1_embed_dim)
            
        self.encoder = Encoder_v05(
            layer_io=layer_io,
            c1_embed_dim = c1_embed_dim, 
            layer_norm=layer_norm,
            inject_c1_eachlayer=inject_c1_eachlayer,
        )
        
        self.decoder = Decoder_v05(
            layer_io=layer_io[::-1], # reverse the encoder 
            c1_embed_dim=c1_embed_dim,
            layer_norm=layer_norm,
            sigmoid_out=sigmoid_out,
        )
        
    def forward(self, x, c1=None):
        if c1 is not None:
            c1 = self.c1_embedding(c1)
            z = self.encoder(x, c1)
            xhat = self.decoder(torch.cat((z, c1), dim=-1))
        else:
            z = self.encoder(x)
            xhat = self.decoder(z)
        if self.return_latent:
            return xhat, z
        else:
            return xhat
        
# cVAEv03

class cVAE(nn.Module):
    def __init__(self, 
                 feat_in : int = 50000, 
                 n_hidden : int = 256,
                 enc_feat_out : int = 64,
                 nb_layers: int = 4,
                 n_c1_class : int = 58,
                 c1_embed_dim : int = 8,
                 inject_c1_eachlayer: bool = False,
                 sigmoid_out: bool = False,
                 return_latent : bool = False):
        super().__init__()
        self.return_latent = return_latent
        
        if c1_embed_dim != 0 and n_c1_class != 0:
            self.c1_embedding = nn.Embedding(n_c1_class, c1_embed_dim)
        
        self.encoder = Encoder(
            feat_in=feat_in, 
            n_hidden=n_hidden,
            feat_out=enc_feat_out,
            nb_layers=nb_layers,
            inject_c1_eachlayer=inject_c1_eachlayer,
            # embed_dim=embed_dim,
        )

        self.calc_mean = nn.Sequential(
            FC(I=enc_feat_out + c1_embed_dim, O=enc_feat_out, 
               norm=nn.LayerNorm(enc_feat_out, elementwise_affine=False),
               D=0.1, name='mean1'),
            nn.Linear(enc_feat_out, enc_feat_out)
        )
        
        self.calc_logvar = nn.Sequential(
            FC(I=enc_feat_out + c1_embed_dim, O=enc_feat_out, 
               norm=nn.LayerNorm(enc_feat_out, elementwise_affine=False),
               D=0.1, name='logvar1'),
            nn.Linear(enc_feat_out, enc_feat_out)
        )
        
        self.decoder = Decoder(
            feat_in=enc_feat_out,
            n_hidden=n_hidden,
            feat_out=feat_in,
            nb_layers=nb_layers,
            c1_embed_dim=c1_embed_dim,
            sigmoid_out=sigmoid_out,
        )
        
    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(mean.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma
    
    def forward(self, x, c1=None):
        x = self.encoder(x)
        if c1 is not None:
            c1 = self.c1_embedding(c1)
            mean = self.calc_mean(torch.cat((x, c1), dim=-1))
            logvar = self.calc_logvar(torch.cat((x, c1), dim=-1))
        else:
            mean, logvar = self.calc_mean(x), self.calc_logvar(x)
        z = self.sampling(mean, logvar)
        if self.return_latent:
            return self.decoder(z, c1), mean, logvar, z
        else:
            return self.decoder(z, c1), mean, logvar
    
    def generate(self, class_idx, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        if (type(class_idx) is int):
            class_idx = torch.tensor(class_idx)
        class_idx = class_idx.to(device)
        if (len(class_idx.shape) == 0):
            batch_size = None
            class_idx = class_idx.unsqueeze(0)
            z = torch.randn((1, self.dim)).to(device)
        else:
            batch_size = class_idx.shape[0]
            z = torch.randn((batch_size, self.dim)).to(device) 
        y = self.label_embedding(class_idx)
        res = self.decoder(z, y)
        if not batch_size:
            res = res.squeeze(0)
        return res
        