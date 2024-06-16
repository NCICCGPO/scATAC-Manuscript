"""
Assume that the model state dict is properly loaded
"""


import os
import sys
import time
import datetime
import numpy as np
from collections import OrderedDict
import tqdm
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from sklearn.decomposition import PCA
import pickle

import sys
sys.path.append('/project/')
from drscax.scripts import models as tcgamodels
from drscax.scripts import data as tcgadata

mfp = 'model_zoo/dev/'
dfp = 'data/'

class eval_model():
    def __init__(self, 
                 dataloader, 
                 md,
                 model, 
                 log,
                 approach: str = '1', 
                 nocondition: bool = False,
                 model_name: str = 'cVAE',
                 save_name: str = None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 pfp='/project/drscax/results'
                ):
        """
        Evaluate models

        Arguments:
          **dataloader** torch.utils.data.Dataloader
          **md** sc.AnnData.obs
            metadata as pd.DataFrame which is usually stored in the adata.obs slot of AnnDatas
          **save_name** str (default: `None`)
            where to save all the stuff to 
        """
        self.dl = dataloader
        self.md = md
        self.model = model
        self.log = log
        self.approach = approach
        self.nocondition = nocondition
        self.model_name = model_name
        self.save_name = save_name
        self.device = device
        self.pfp = pfp
        self.save_shortname = os.path.split(save_name)[1].split('.pt')[0]

    def agg_data(self):
        """
        Aggregate data into one tensor.

        Returns: 
          **data** dict 
        """
        N = self.dl.dataset.__len__()

        barcodes = []

        self.model.eval()
        count = 0
        for i, batch in enumerate(tqdm.tqdm(self.dl)):
            x, y, x_div_cnv, idx = batch
            x, y, x_div_cnv = x.to(self.device), y.to(self.device), x_div_cnv.to(self.device)
            n = x.shape[0]
            
            if self.approach == '1':
                if self.nocondition:
                    output = self.model(x_div_cnv)
                else:
                    output = self.model(x_div_cnv, y)
            elif self.approach == '2':
                if self.nocondition:
                    output = self.model(x)
                else:
                    output = self.model(x, y)
                    
            if 'VAE' in self.model_name:
                xhat, mean, logvar, z = output
            else:
                xhat, z = output
            del output
            barcodes += idx
            if i==0:
                X = torch.empty(N, x.shape[1])
                # X_div_cnv = torch.empty(N, x_div_cnv.shape[1])
                Xhat = torch.empty(N, xhat.shape[1])
                Z = torch.empty(N, z.shape[1])
            X[count:count+n] = x.detach().cpu()
            # X_div_cnv[count:count+n] = x_div_cnv.detach().cpu()
            Xhat[count:count+n] = xhat.detach().cpu()
            Z[count:count+n] = z.detach().cpu()
            count += n
        return X, Xhat, Z, barcodes
    
    def agg_tcga(self):
        """
        Aggregate data into one tensor.

        Returns: 
          **data** dict 
        """
        N = self.dl.dataset.__len__()

        barcodes = []

        self.model.eval()
        count = 0
        for i, batch in enumerate(tqdm.tqdm(self.dl)):
            if self.dl.dataset.split == 'all' or self.dl.dataset.split == 'test':
                x, y, idx = batch
            else:
                x, y, x_div_cnv, idx = batch
                x_div_cnv = x_div_cnv.to(self.device)
            x, y = x.to(self.device), y.to(self.device)
            n = x.shape[0]
            
            if self.approach == '1':
                if self.nocondition:
                    output = self.model(x_div_cnv)
                else:
                    output = self.model(x_div_cnv, y)
            elif self.approach == '2':
                if self.nocondition:
                    output = self.model(x)
                else:
                    output = self.model(x, y)
                    
            if 'VAE' in self.model_name:
                xhat, mean, logvar, z = output
            else:
                xhat, z = output
            del output
            barcodes += idx
            if i==0:
                X = torch.empty(N, x.shape[1])
                # X_div_cnv = torch.empty(N, x_div_cnv.shape[1])
                Xhat = torch.empty(N, xhat.shape[1])
                Z = torch.empty(N, z.shape[1])
            X[count:count+n] = x.detach().cpu()
            # X_div_cnv[count:count+n] = x_div_cnv.detach().cpu()
            Xhat[count:count+n] = xhat.detach().cpu()
            Z[count:count+n] = z.detach().cpu()
            count += n
        return X, Xhat, Z, barcodes

    def viz_umap(self, mat, barcodes, use_pca=False, plot_file=None, include_leiden=True):
        # make adata for approach 1
        tdata = sc.AnnData(X=mat.numpy(), obs=self.md.loc[barcodes, :])
        if 'tissue' not in tdata.obs.columns:
            tdata.obs['tissue'] = tdata.obs['Sample'].apply(lambda s: s.split('_')[1])
        if 'n_idx' not in tdata.obs.columns:
            # add a numerical index
            tdata.obs['n_idx'] = np.arange(tdata.shape[0])
        if use_pca:
            pca = PCA(n_components=30)
            pca.fit(tdata.X)
            tdata.obsm['X_pca'] = pca.transform(tdata.X)
            sc.pp.neighbors(tdata, n_pcs=30)
        else:
            sc.pp.neighbors(tdata, n_pcs=0, use_rep=None) # use .X
        if include_leiden:
            sc.tl.leiden(tdata)
        sc.tl.umap(tdata)
        if plot_file is not None:
            sc.settings.figdir, save_name = os.path.split(plot_file)
        else:
            save_name = None
        if include_leiden:
            colors = ['batch', 'tissue', 'leiden']
        else:
            colors = ['batch', 'tissue']
        sc.pl.umap(tdata, color=colors,
                   save=save_name)
        return tdata

    def clustering_metrics(self,
                           labels_true=[0, 0, 1, 1, 2, 2, 3, 3,],
                           labels_pred=[0, 0, 0, 1, 2, 3, 4, 5],
                           verbose=True):
        homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
        ARI = metrics.adjusted_rand_score(labels_true, labels_pred)

        if verbose:
            print('  ARI: {:.3f}\tv-measure: {:.3f}\thomogeneity: {:.3f}\tcompleteness: {:.3f}'.format(
                ARI,
                v_measure,
                homogeneity,
                completeness,
            ))
        return ARI, v_measure, homogeneity, completeness


    def reconstruction_plots(self, X, Xhat, save_plot=False, sample=False):
        if sample:
            idx_x = np.random.choice(np.arange(X.shape[0]), 128, replace=True)
            idx_y = np.random.choice(np.arange(X.shape[1]), 1024, replace=True)
            X = X[idx_x, :]
            X = X[:, idx_y]
            Xhat = Xhat[idx_x, :]
            Xhat = Xhat[:, idx_y]

        p0 = sns.clustermap(
            X.detach().cpu(), 
            row_cluster=False, 
            col_cluster=False,
            # row_colors=adata.obs.loc[list(idx), 'tissue'].to_numpy(),
            annot=False, 
            cmap="RdYlBu_r",
            # linecolor='b', 
            cbar=True,
            xticklabels=False,
            yticklabels=False,
            # ax=ax[0],
        )
        p0.fig.suptitle('X')

        # p1 = sns.clustermap(
        #     X_div_cnv.detach().cpu(), 
        #     row_cluster=False, 
        #     col_cluster=False,
        #     # row_colors=adata.obs.loc[list(idx), 'tissue'].to_numpy(),
        #     annot=False, 
        #     cmap="RdYlBu_r",
        #     # linecolor='b', 
        #     cbar=True,
        #     xticklabels=False,
        #     yticklabels=False,
        #     # ax=ax[0],
        # )
        # p1.fig.suptitle('X_div_cnv')

        p2 = sns.clustermap(
            Xhat.detach().cpu(), 
            row_cluster=False, 
            col_cluster=False,
            # row_colors=adata.obs.loc[list(idx), 'tissue'].to_numpy(),
            annot=False, 
            cmap="RdYlBu_r",
            # linecolor='b', 
            cbar=True,
            xticklabels=False,
            yticklabels=False, 
            # ax=ax[1],
        )   
        p2.fig.suptitle('Xhat')
        
        if save_plot:
            p0.savefig(os.path.join(self.pfp, 'clustermap_X_{}.png'.format(self.save_shortname)), 
                           bbox_inches='tight')
                # p1.savefig('/home/nravindra/project/drscax/results/clustermap_X_div_cnv_{}.png'.format(os.path.split(save_name)[1].split('.pt')[0]), 
                #            bbox_inches='tight')
            p2.savefig(os.path.join(self.pfp, 'clustermap_Xhat_{}.png'.format(self.save_shortname)), 
                           bbox_inches='tight')
                       
        return None

    def loss_curve(self, save_plot=False):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

        sns.lineplot(x=np.arange(1, len(self.log['<loss_train>'])+1), 
                     y=self.log['<loss_train>'], label='train', 
                     ax=ax)
        sns.lineplot(x=np.arange(1, len(self.log['<loss_val>'])+1), 
                     y=self.log['<loss_train>'], label='val',
                     ax=ax)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('<loss>_sample')
        ax.set_title(self.save_shortname)
        if save_plot:
            fig.savefig(os.path.join(self.pfp, 'lineplot_loss_{}.png'.format(self.save_shortname)), 
                        bbox_inches='tight')

        return None
    
    def custom_metric(self, tdata, n_neighbors=100, n_samples=1000, save_plot=False):
        """
        Arguments: 
          **tdata** sc.AnnData
            NOTE: tdata.pp.neighbors will be recalculated, since different from default umap viz 
        """
        
        sc.pp.neighbors(tdata, 
                        n_pcs=0, 
                        n_neighbors=n_neighbors, 
                        use_rep=None)
        
        # idx = tdata.obs.loc[tdata.obs['tissue']==tissue, :].sample(n_samples, replace=True).index
        idx = tdata.obs.sample(n_samples, replace=True).index
        tmp = pd.DataFrame(tdata.obs.loc[idx, 'n_idx'].apply(lambda x: np.where((tdata.obsp['connectivities'][x, :] > 0).toarray())[1]))
        tmp['j_batch'] = tmp['n_idx'].apply(lambda x: [tdata.obs.loc[tdata.obs.index[i], 'batch'] for i in x])
        tmp['j_tissue'] = tmp['n_idx'].apply(lambda x: [tdata.obs.loc[tdata.obs.index[i], 'tissue'] for i in x])
        tmp['i_batch'] = tdata.obs.loc[tmp.index, 'batch']
        tmp['i_tissue'] = tdata.obs.loc[tmp.index, 'tissue']
        tmp['same_batch_same_tissue'] = tmp.apply(lambda x: sum([True if x['i_batch']==ib and x['i_tissue']==it else False for ib, it in zip(x['j_batch'], x['j_tissue'])])/len(x['j_batch']), axis=1)
        tmp['diff_batch_same_tissue'] = tmp.apply(lambda x: sum([True if x['i_batch']!=ib and x['i_tissue']==it else False for ib, it in zip(x['j_batch'], x['j_tissue'])])/len(x['j_batch']), axis=1)
        tmp['diff_batch_diff_tissue'] = tmp.apply(lambda x: sum([True if x['i_batch']!=ib and x['i_tissue']!=it else False for ib, it in zip(x['j_batch'], x['j_tissue'])])/len(x['j_batch']), axis=1)
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 2))
        sns.distplot(tmp['same_batch_same_tissue'], label='same_batch_same_tissue', ax=ax)
        sns.distplot(tmp['diff_batch_same_tissue'], label='diff_batch_same_tissue', ax=ax)
        sns.distplot(tmp['diff_batch_diff_tissue'], label='diff_batch_diff_tissue', ax=ax)
        ax.legend().set_visible(True)
        
        if save_plot:
            fig.savefig(os.path.join(self.pfp, 'distplot_custom_knn_metric_{}.png'.format(self.save_shortname)), 
                        bbox_inches='tight')
        
        return tmp['same_batch_same_tissue'], tmp['diff_batch_same_tissue'], tmp['diff_batch_diff_tissue'] 
    
    def eval_all(self):
        X, Xhat, Z, barcodes = self.agg_tcga()
        tdata = self.viz_umap(Z, barcodes, plot_file=os.path.join(self.pfp, self.save_shortname + '.png'))
        ttdata = self.viz_umap(Z, barcodes, use_pca=True, plot_file=os.path.join(self.pfp, self.save_shortname + '_Zto30dimPCA.png'))
        ttdata = self.viz_umap(Xhat, barcodes, use_pca=True, plot_file=os.path.join(self.pfp, self.save_shortname + '_Xhatto30dimPCA.png'))
        ARI_batch, V_batch, homo_batch, comp_batch = self.clustering_metrics(labels_true=tdata.obs['batch'],
                                                                        labels_pred=tdata.obs['leiden'])
        ARI_tissue, V_tissue, homo_tissue, comp_tissue = self.clustering_metrics(labels_true=tdata.obs['tissue'],
                                                                        labels_pred=tdata.obs['leiden'])
        self.reconstruction_plots(X, Xhat, save_plot=True, sample=True)
        self.loss_curve(save_plot=True)
        sbst, dbst, dbdt = self.custom_metric(tdata, save_plot=True)
        
        # add metrics to log
        self.log['ARI_batch'] = ARI_batch
        self.log['V_batch'] = V_batch
        self.log['ARI_tissue'] = ARI_tissue
        self.log['V_tissue'] = V_tissue
        self.log['sbst'] = sbst
        self.log['dbst'] = dbst
        self.log['dbdt'] = dbdt
        return self.log
            

# standalone
def viz_umap(mat, barcodes, md, use_pca=False, plot_file=None, include_leiden=True):
    from sklearn.decomposition import PCA
    # make adata for approach 1
    tdata = sc.AnnData(X=mat.numpy(), obs=md.loc[barcodes, :])
    if 'tissue' not in tdata.obs.columns:
        tdata.obs['tissue'] = tdata.obs['Sample'].apply(lambda s: s.split('_')[1])
    if 'n_idx' not in tdata.obs.columns:
        # add a numerical index
        tdata.obs['n_idx'] = np.arange(tdata.shape[0])
    if use_pca:
        pca = PCA(n_components=30)
        pca.fit(tdata.X)
        tdata.obsm['X_pca'] = pca.transform(tdata.X)
        sc.pp.neighbors(tdata, n_pcs=30)
    else:
        sc.pp.neighbors(tdata, n_pcs=0, use_rep=None) # use .X
    if include_leiden:
        sc.tl.leiden(tdata)
    sc.tl.umap(tdata)
    if plot_file is not None:
        sc.settings.figdir, save_name = os.path.split(plot_file)
    else:
        save_name = None
    if include_leiden:
        colors = ['batch', 'tissue', 'leiden']
    else:
        colors = ['batch', 'tissue']
    sc.pl.umap(tdata, color=colors,
               save=save_name)
    return tdata

def gen_npoutput(AnnData, model, 
                 return_ZtoPCA_adata=True, # this is the best so far
                 exp_name='bst8layer25k_woimmune',
                 save_path='/tmp/', 
                 chunk_size=1024,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    print('Loading data...')
    tic = time.time()
    
    # assume full adata wanted
    exp_name = '{}_{}'.format(exp_name, datetime.datetime.today().strftime('%y%m%d_%H%M%S'))
    N = AnnData.shape[0]
    barcodes = []

    model.eval()
    count = 0
    for i in tqdm.tqdm(range(0, N, chunk_size)):
        x = torch.tensor(AnnData.X[i:i+chunk_size].toarray(), dtype=torch.float32)
        x = x.to(device)
        n = x.shape[0]

        xhat, z = model(x)

        barcodes += AnnData.obs.iloc[i:i+chunk_size].index.to_list()

        if i==0:
            X = torch.empty(N, x.shape[1])
            # X_div_cnv = torch.empty(N, x_div_cnv.shape[1])
            Xhat = torch.empty(N, xhat.shape[1])
            Z = torch.empty(N, z.shape[1])
        X[count:count+n] = x.detach().cpu()
        # X_div_cnv[count:count+n] = x_div_cnv.detach().cpu()
        Xhat[count:count+n] = xhat.detach().cpu()
        Z[count:count+n] = z.detach().cpu()
        count += n
        
    print('\n  generating UMAPs. sec elapsed: {:.0f}'.format(time.time() - tic))
    if save_path is not None:
        plot_file1 = os.path.join(save_path, 'Z_{}.png'.format(exp_name))
        plot_file2 = os.path.join(save_path, 'Zto30dPCA_{}.png'.format(exp_name))
    else:
        plot_file1, plot_file2 = None, None
    tdata_Z = viz_umap(Z, barcodes, AnnData.obs, use_pca=False, plot_file=plot_file1, include_leiden=False)
    tdata_Z_PCA = viz_umap(Z, barcodes, AnnData.obs, use_pca=True, plot_file=plot_file2, include_leiden=False)
        
    if save_path is not None:
        print('\n  saving. sec elapsed: {:.0f}'.format(time.time() - tic))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        np.save(os.path.join(save_path, 'X_{}.npy'.format(exp_name)), X.numpy())
        np.save(os.path.join(save_path, 'Xhat_{}.npy'.format(exp_name)), Xhat.numpy())
        np.save(os.path.join(save_path, 'Z_{}.npy'.format(exp_name)), Z.numpy())
        np.save(os.path.join(save_path, 'barcodes_{}.npy'.format(exp_name)), np.array(barcodes))
        np.save(os.path.join(save_path, 'Z_umap_{}.npy'.format(exp_name)), tdata_Z.obsm['X_umap'])
        np.save(os.path.join(save_path, 'Zto30dPCA_umap_{}.npy'.format(exp_name)), tdata_Z_PCA.obsm['X_umap'])
        
        
    print('\n  DONE. sec elapsed: {:.0f}'.format(time.time() - tic))
    
    if return_ZtoPCA_adata:
        return tdata_Z_PCA
    else:
        return tdata_Z

