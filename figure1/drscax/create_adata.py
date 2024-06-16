"""
Data pipeline:
  1. run `rds2_mtx_csv.ipynb` which takes all rds data files and outputs to 
       scipy.sparse mtx files and stores cellular metadata 
  2. run `tcga().create_adata(out_file=*date.h5ad)
       *NOTE*: this can be run in create_adata_dev.ipynb dynamically

Remainder of data pipeline in experiment implementation
"""


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

import scanpy as sc

    
class rds_csv_2_h5ad():
    def __init__(self,
                 peak_md_filter: str = './peak_metadata.csv',
                 md_fp: str = './barcodes/', 
                 mtx_fp: str = './sparsemtx/',
                 cnv_fp: str = './TCGA/Synapse/cleaned_AWS/',):
        """
        Assume that preprocessing has been done and the cell x peak counts matrix is in a .rds
          file.
          
        Args:
          **peak_md_high_cnt** str (default: `~/peak_metadata_220921.csv'`)
            A csv indexed per peak that has its rank and total counts accross the [training] dataset
          
        """
        self.peak_md_filter = peak_md_filter
        self.md_fp = md_fp
        self.mtx_fp = mtx_fp
        self.cnv_fp = cnv_fp
        
    
    def sample_cnv(self, peak_md, sample_copynb_file, overlap_fraction=0.8):
        '''Output the sample copy number.

        Arguments:
          peak_metadata (pd.DataFrame)
          overlap_fraction (optional, float): see -f argument in bedtools.intersect
        '''
        import pybedtools
        A = pybedtools.BedTool.from_dataframe(peak_md[['seqnames', 'start', 'end']])
        copy_nb = pd.read_csv(sample_copynb_file)
        B = pybedtools.BedTool.from_dataframe(copy_nb[['Chromosome', 'Start', 'End', 'Copy_Number']])
        # overlap = B.intersect(A, wo=True, F=0.8).to_dataframe()
        overlap = B.intersect(A, wo=True).to_dataframe() #todo: add overlap_fraction arg after understanding what it does & how to trigger it

        # clean up and merge metadata; x=metadata, y=overlap
        peak_md_with_cnv = peak_md.merge(overlap, left_on=['seqnames', 'start', 'end'], right_on=['score', 'strand', 'thickStart'], how='left')
        peak_md_with_cnv['name'] = peak_md_with_cnv['name'].fillna(0)
        peak_md_with_cnv = peak_md_with_cnv.rename(columns={'name': 'copy_nb'})
        return peak_md_with_cnv
    
    def term_freq(self, x, verbose=False):
        '''
        Arguments:
          x (scipy.sparse.csr_matrix): a cell x peak_counts matrix where terms are peaks (columns)

        '''
        if verbose:
            print('x.type:', type(x))
            print('x.sum.type:', type(x.sum(axis=1)))
        tf = x / x.sum(axis=1)
        if verbose:
            print('tf.type:', type(tf))
            # all rows summed should be equal to one
            print('Worked (rows sum to 1):', tf.sum(axis=1))
                  # (tf.sum(axis=1) == np.ones((tf.shape[0],)))).all()
        return np.asarray(tf)


    def document_freq(self, x, verbose=False):
        '''The number of docs (cells) in which the term (peak) is present.

        NOTE: 
          - DF is the nb of docs in which the word is present, we consider one occurence if the term is present
            in the doc at least once, we don't need the nb of times the term is present
        Arguments:
          x (scipy.sparse.csr_matrix or np.array): cell x peak_counts matrix
        '''
        return (x > 0.).astype(np.float32).sum(axis=0) / x.shape[0]

    def inv_doc_freq(self, x, log_idf=True):
        '''Inverse document frequency.

        NOTE:
          - with large corpus (N_cells), IDF can explode so
            take log of it and avoid division by zero
        '''
        if log_idf:
            idf = np.log(x.shape[0] / (self.document_freq(x) + 1))
        else: 
            idf = x.shape[0] / (self.document_freq(x) + 1)
        return np.asarray(idf).squeeze()

    def tfidf(self, x, log_idf=True):
        '''term_frequency * inverse_document_frequency where cells are documents, peaks are terms, and the corpus is a sample single-cell dataset

        NOTE:
          - idf is log pseudocount and a term is counted as occuring in a document (peak in cell) if count is gt 0 

        Arguments:
          x (scipy.sparse.csr_matrix): a cell x peak_counts matrix where terms are peaks (columns)

        '''
        return self.term_freq(x) * self.inv_doc_freq(x, log_idf=log_idf)
    
    def create_adata(self, topn=50000, calc_tfidf=True, out_file='~/<outfilename>.h5ad', verbose=False):
        peak_md = pd.read_csv(self.peak_md_filter, index_col=0)
        peak_md = peak_md.reset_index()
        col_idx = peak_md.loc[peak_md['cnt_rank'] < topn, :].index.to_list()
        
        adatas = {}
        peaks_with_cnv = {}
        for i, f in tqdm.tqdm(enumerate(glob.glob(os.path.join(self.mtx_fp, '*.mtx')))):
            assert os.path.exists(f), 'sparse .mtx not found'
            # cell x peak counts matrix
            mat = io.mmread(f)
            mat = mat.astype(np.float32).transpose()
            mat = sparse.csr_matrix(mat)[:, col_idx]
            if verbose:
                print('  mat.shape:', mat.shape)
            
            # cell metadata
            basename = os.path.split(f)[1].split('.mtx')[0]
            if verbose:
                print('\n  processing file:', basename)
            md_file = os.path.join(self.md_fp, basename + '.csv')
            assert os.path.exists(md_file), 'cell metadata .csv not found'
            md = pd.read_csv(md_file, index_col=0)
            
            # peak metadata
            cnv_file = os.path.join(self.cnv_fp, 
                            '_'.join(basename.split('_')[2:7]) + '.csv')
            if os.path.exists(cnv_file):
                peak_md_cnv = self.sample_cnv(peak_md, cnv_file)
            else:
                peak_md_cnv = peak_md.copy()
            peak_md_cnv = peak_md_cnv.iloc[col_idx, :]
            peak_md_cnv = peak_md_cnv.reset_index()
            peak_md_cnv.index = peak_md_cnv.index.astype(str)
            if verbose:
                print('  peak_md_cnv.shape:', peak_md_cnv.shape)
            adata = sc.AnnData(X=mat, obs=md, var=peak_md_cnv)
            if os.path.exists(cnv_file):
                peaks_with_cnv[i] = peak_md_cnv # save for later
            del mat, peak_md_cnv
            if calc_tfidf:
                adata.X = sparse.csr_matrix(np.log(self.tfidf(adata.X, log_idf=False)*1e4 + 1)) 
            adata.obs['batch'] = i # add label
            adata.obs['split'] = 'train' if os.path.exists(cnv_file) else 'test'
            adatas[i] = adata
            if verbose:
                print(adata)
                print('')
            del adata
            
        # save pickle
        if True and out_file is not None:
            f1, f2 = os.path.split(out_file)
            f2 = f2.split('.h5ad')[0]
            with open(os.path.join(f1, f2 + '.pkl'), 'wb') as f3:
                pickle.dump(adatas, f3, protocol=pickle.HIGHEST_PROTOCOL)
                f3.close()
                
        if True and out_file is not None:
            # save peaks
            f1, _ = os.path.split(out_file)
            f2 = 'peaks_with_cnv_dict.pkl'
            with open(os.path.join(f1, f2), 'wb') as f3:
                pickle.dump(peaks_with_cnv, f3, protocol=pickle.HIGHEST_PROTOCOL)
                f3.close()
                
        # merge
        print('\nmerging...')
        adatas = sc.concat(adatas, label='batch_id') # obliterates metadata
        print('\n  merged adata:')
        print(adatas)
        
        if out_file is not None:
            adatas.write(out_file)
        
        ## add in the metadata to the var slot, then save    
        for i, k in enumerate(peaks_with_cnv.keys()):
            if i==0:
                adatas.var = peaks_with_cnv[k].rename(columns={
                    'copy_nb': 'copy_nb_batch{}'.format(k)
                    })
            else:
                adatas.var = adatas.var.merge(peaks_with_cnv[k].rename(columns={
                    'copy_nb': 'copy_nb_batch{}'.format(k)}).loc[:, 
                                                                 ['index', 'copy_nb_batch{}'.format(k)]], 
                                      how='left',
                        left_on='index', right_on='index')
                
        # resave the adata.var with 
        if out_file is not None:
            adatas.write(out_file)
        return adatas
        
if __name__ == '__main__':
    tic = time.time()
    print('Creating data...')
    dfp = '/h5ad'
    out_file = os.path.join(dfp, '<outfilename>.h5ad')
    data_creator = rds_csv_2_h5ad()
    adatas = data_creator.create_adata(topn=50000, calc_tfidf=True, out_file=out_file, verbose=True)
    print(adatas)
    print('\n  done after {:.1f}-min'.format((time.time() - tic)/60))
    print('  adata dumped at:', out_file)
