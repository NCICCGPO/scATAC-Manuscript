"""
Instructions:
  1. 

TODO: 
  - optimize namespace to simplify calls to the class
"""

import pandas as pd

import sys
sys.path.append('/./')
import reg_diffs.scripts.models as regdiffmodels
import reg_diffs.scripts.data as regdiffdata

import re
import os
import numpy as np
import gc
import pandas as pd
import time
import tensorflow as tf
import glob
import datetime
import itertools
import glob

class ism(object):
    def __init__(
        self,
        file: str or None = None,
        file_specifier: str = None, # need this for sub routine specification of weights
        tmp_dir: str = './tmp/ism/data/',
        mutseq_col_key: str or None = 'Tumor_Seq_Allele2', # if none, ref seq used
        chunksize: int or None = None,
        n_shuffles: int = 6,
        n_flank: int = 5, 
        output_length: int = 1364,
        chrom_key: str = 'Chromosome',
        start_key: str = 'hg38_start',
        end_key: str = 'hg38_end',
        id_key: str = 'mutation_id',
        script: str = '/./reg_diffs/experiments/glism.sh',
        verbose: bool = True,
    ):
        """
        TODO:
          1. add easier functionality for adding valid, new file_specifier(s)

        Arguments:
          file_specifier is the current way to trigger path to weights. VALID is 'brca', 'blca', and so on
        """
        # assert file_specifier is not None or file is not None, 'specify hard coded files via file_specifier or pt to gene loc dataframe file'
        # assert path_to_weights is not None if file_specifier is None else True, 'if not hard coding file_specifier, e.g., sample, add path to weights'
        self.file = file
        self.file_specifier = file_specifier
        self.tmp_dir = tmp_dir
        self.mutseq_col_key = mutseq_col_key
        self.chunksize = chunksize
        self.n_shuffles = n_shuffles
        self.n_flank = n_flank
        self.output_length = output_length
        self.chrom_key = chrom_key
        self.start_key = start_key
        self.end_key = end_key
        self.id_key = id_key
        self.script = script
        self.verbose = verbose
        assert output_length >= 2*n_flank + 1, 'output seq len must be larger than flank in current implementation'


    @staticmethod
    def get_seq(
        fasta_seq,
        mut_seq_at_start: str or None = None,
        n_flank: int = 3, 
        chrom: str = 'chr1',
        start: int = 1,
        end: int = None, 
        output_length: int or None = None,
        return_indices: bool = False,
    ):
        """
        NOTE: this is biased to be right-shifted by an additional base-pair.
              stochastic padding removed to always add 1 to RHS of seq 
              to yield predictable behavior if the segment is of odd length

        Arguments:
          mut_seq_at_start: str or None (optional, Default: None)
            NOTE: this assumes that the mutation value fed here begins at the
                  start seq 
        """            
        L = start - n_flank
        R = end + n_flank 
        L_seqloc = 0
        R_seqloc = R - L 

        if output_length is not None:
            pad = 1 if (output_length - R_seqloc) % 2 == 1 else 0
            L = L - (output_length - R_seqloc)//2 # can randomize with - (1-pad) and + pad for np.random.choice(0, 1) if OL not even
            R = R + (output_length - R_seqloc)//2 + pad
            assert R - L == output_length, 'wrong indexing. R: {}\tL: {}\t output_length: {}'.format(R, L, output_length)
            if return_indices:
                L_seqloc = L_seqloc + (output_length - R_seqloc)//2
                R_seqloc = L_seqloc + R_seqloc
                # print('{}:{}'.format(L_seqloc, R_seqloc))
        
        if mut_seq_at_start is not None:
            seq = str(fasta_seq[chrom][L:start]) + mut_seq_at_start + str(fasta_seq[chrom][start+1:R])
        else: 
            seq = str(fasta_seq[chrom][L:R])

        if return_indices:
            return seq, (L_seqloc, R_seqloc)
        else:
            return seq

    @staticmethod
    def filter_nonACGT(
        df: pd.DataFrame, 
        n_flank: int or None = 5,
        output_length: int or None = None, 
        chrom_key: str = 'Chromosome',
        start_key: str = 'hg38_start',
        end_key: str = 'hg38_end',
        vocab: list = ['A', 'C', 'G', 'T'],
    ):  
        if output_length is None:
            assert n_flank is not None, 'since not creating output_length, specify n_flank'
        if n_flank is None:
            assert output_length is not None, 'since not adding n_flank, specify output_length'
        if n_flank is not None and output_length is not None:
            assert True, 'specify mode with n_flank OR output_length, other as None'
        counter = 0
        fasta_seq = regdiffdata.get_fasta()
        print('df shape before non-ACGT chk:', df.shape)
        vocab.sort()
        ACGT_only = []
        for i, r in df[[chrom_key, start_key, end_key]].iterrows():
            if n_flank is None:
                seq = str(fasta_seq[r[chrom_key]][r[start_key] - output_length//2 - 1:r[end_key] + output_length//2 + 1])
            elif output_length is None:
                seq = str(fasta_seq[r[chrom_key]][r[start_key] - n_flank:r[end_key] + n_flank + 1])
            seq = set(seq)
            all_ACGT = all([False if s not in vocab else True for s in seq])
            ACGT_only.append(all_ACGT)
            if not all_ACGT:
                print('Problem in {}\t{}\t{}\tset(seq):'.format(r[chrom_key], r[start_key], r[end_key]), seq)
                counter += 1
        # filter
        df = df.loc[ACGT_only, :]
        print('df shape after non-ACGT chk:', df.shape)
        print('{} bad rows'.format(counter))
        # del df['ACGT_only']
        return df



            

    def pp_chunk(self, chunk: pd.DataFrame,):
        """create chunk files to later be loaded by glob and processed

        Arguments:
        file: str
            csv file of motif_name, chrom, start, end columns
        chunksize: int or None
            How many motifs/peaks to go through per job. If "None", don't chunk
        path_to_weights: str
            use the brackets "{}" to indicate the fold OR, if no brackets, assume it's just one model needed
        window_size: int (optional, default: 20)
            a symmetrical window about the peak midpoint. Must be an even number
        output_length: int (optional, default: 1364)
            specify the model size and input sequence length, which is useful for picking midpt as well as this//2
        batch_size: int (optional, default: 128)
            note, this will get divided by n_shuffles + 1 because this will go through and generate the model output 
            per seq, which will in turn have n_shuffles x sequences
        n_folds: int (optional, default: 5)
            number of state dicts to load for the model
        output_path: str (optional, default: '~/brca_v01_motifism_{}.pkl')
            these can be aggregated later
        
        Returns:
        results: dict
            aggregate of results
        """
        fasta_seq = regdiffdata.get_fasta()

        # add seqs
        chunk['seq'] = [
            self.get_seq(
                fasta_seq=fasta_seq, 
                mut_seq_at_start=r[self.mutseq_col_key] if self.mutseq_col_key is not None else None,
                chrom=r[self.chrom_key],
                start=r[self.start_key],
                end=r[self.end_key],
                n_flank=self.n_flank,
                output_length=self.output_length,
            ) for i, r in chunk.iterrows()]
        
        if self.output_length is None:
            chunk['shuffled_seqs'] = chunk['seq'].apply(lambda x: np.hstack([regdiffdata.dinuclShuffle(x) for i in range(self.n_shuffles)]))
        else:
            # calculate L:segment:R
            L_seq = chunk[self.end_key] - chunk[self.start_key] + 2*self.n_flank
            L = (self.output_length - L_seq) // 2
            R = L + L_seq
            shufled_seqs = []
            assert L.shape[0] == R.shape[0] and R.shape[0] == chunk.shape[0], 'shape mismatch in L:R code'
            shuffled_seqs = []
            for i in range(R.shape[0]):
                Lflank = chunk['seq'].iloc[i][:L.iloc[i]]
                segment = chunk['seq'].iloc[i][L.iloc[i]:R.iloc[i]]
                Rflank = chunk['seq'].iloc[i][R.iloc[i]:]
                shuffled_seqs.append(
                    np.hstack([Lflank + regdiffdata.dinuclShuffle(segment) + Rflank for i in range(self.n_shuffles)])
                )

            chunk['shuffled_seqs'] = shuffled_seqs
            del shuffled_seqs
            
        # flatten chunk
        id_cols = [self.chrom_key, self.start_key, self.end_key, self.id_key]
        chunk = pd.melt(chunk.loc[:, id_cols + ['seq', 'shuffled_seqs']], 
                        id_vars=id_cols, 
                        var_name='source', value_name='seqs')
        chunk = chunk.merge(
            chunk['seqs'].apply(pd.Series).stack().rename('seqs').reset_index(),
            right_on='level_0', left_index=True, suffixes=(['_old', '']))
        chunk = chunk[id_cols + ['source', 'seqs']]

        return chunk
            
    @staticmethod
    def get_model_output(
        chunk_file: str or pd.DataFrame,
        path_to_weights: str,
        batch_size: int = 128,
        n_folds: int = 5,
        output_dir: str or None = None,
        clean_mem: bool = True,
        store_seqs: bool = False,
        verbose: bool = True,
    ):
        """in silico mutagenesis for file, in chunks, specified by name and location
        
        NOTE:
        - this fx is not friendly to custom models. Will have to follow the paths
            to modify model parameters since many assumptions are made here for speed
            of development
        - 
        
        Arguments:
        path_to_weights: str
            use the brackets "{}" to indicate the fold OR, if no brackets, assume it's just one model needed
        window_size: int (optional, default: 20)
            a symmetrical window about the peak midpoint. Must be an even number
        output_length: int (optional, default: 1364)
            specify the model size and input sequence length, which is useful for picking midpt as well as this//2
        batch_size: int (optional, default: 128)
            note, this will get divided by n_shuffles + 1 because this will go through and generate the model output 
            per seq, which will in turn have n_shuffles x sequences
        n_folds: int (optional, default: 5)
            number of state dicts to load for the model
        output_path: str (optional, default: '~/brca_v01_motifism_{}.pkl')
            these can be aggregated later
        
        Returns:
        results: dict
            aggregate of results
        """
        chunk = pd.read_pickle(chunk_file) if not isinstance(chunk_file, pd.DataFrame) else chunk_file
        # now that it is in memory, remove
        if verbose:
            tic = time.time()
            n = chunk.shape[0]
            print('starting model output for chunk:', chunk.shape)
        # iterate through each model, then append width
        if '{}' not in path_to_weights and not isinstance(path_to_weights, list):
            model_state_files = [path_to_weights] # assume 1
            assert n_folds == 1, 'only one path to weights given. Stochasticity or want multiple queries to same file? If so, remove assert.'
            model_state_files = [path_to_weights]*n_folds # will repeat the same model per thingy
        elif isinstance(path_to_weights, list) and all([False if '{}' in i else True for i in path_to_weights]):
            model_state_files = path_to_weights
        else:
            model_state_files = [path_to_weights.format(i) for i in range(n_folds)]
    
        for j, f in enumerate(model_state_files):
            model = regdiffmodels.load_bpnet_chkpt(f) # TODO: only do models.load_weigths to not hog mem and recreate comp graph
        
            # data
            log = np.empty(shape=(chunk.shape[0], ))
            for k, b in enumerate(range(0, chunk.shape[0], batch_size)):
                dt = chunk.iloc[b:b+batch_size, :]
                output = regdiffmodels.get_bpnet_output(regdiffdata.one_hot_encode(dt['seqs']), model)
                log[b:b+batch_size] = output.squeeze()
            chunk['model_{}_out'.format(j)] = list(log)
            # clean 
            del model, output # TODO: this is slow
            tf.keras.backend.clear_session()
            gc.collect()
        
        # clean up 
        if not store_seqs:
            del chunk['seqs'] 

        # output_dir same as input file dir
        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            tmp_name = os.path.split(chunk_file)[1].split('.pkl')[0] if not isinstance(chunk_file, pd.DataFrame) else 'chunk'
            chunk.to_pickle(os.path.join(output_dir, tmp_name + '.pkl'))
            if verbose:
                print('results dumped to: ', os.path.join(output_dir, tmp_name + '.pkl'))
        if output_dir is not None and isinstance(chunk_file, pd.DataFrame):
            print('WARNING. Poor file specification saved because dataframe was input even tho u asked for output_dir')
        if verbose:
            print('through chunk {} in {}-s ({} motifs)'.format(tmp_name, time.time() - tic, n))
        if clean_mem and not isinstance(chunk_file, pd.DataFrame):
            os.remove(chunk_file)
        return chunk


    def chk_chunk_completion(self, ):
        """
        TODO: 
          - the idea here would be to have a flag to resubmit-jobs that aren't completed
            as well as have continuous functionality to check if the eval part of the pipeline can be
            called to round out the script by for example checking unique job names on the server
        """
        return None
    
    def run_ism_pipeline(self,):
        """in silico mutagenesis for file, in chunks, specified by name and location
        
        NOTE:
        - this fx is not friendly to custom models. Will have to follow the paths
            to modify model parameters since many assumptions are made here for speed
            of development
        - 
        
        Arguments:
        motif_file: str
            csv file of motif_name, chrom, start, end columns
        chunksize: int or None
            How many motifs/peaks to go through per job. If "None", don't chunk
        path_to_weights: str
            use the brackets "{}" to indicate the fold OR, if no brackets, assume it's just one model needed
        window_size: int (optional, default: 20)
            a symmetrical window about the peak midpoint. Must be an even number
        output_length: int (optional, default: 1364)
            specify the model size and input sequence length, which is useful for picking midpt as well as this//2
        batch_size: int (optional, default: 128)
            note, this will get divided by n_shuffles + 1 because this will go through and generate the model output 
            per seq, which will in turn have n_shuffles x sequences
        n_folds: int (optional, default: 5)
            number of state dicts to load for the model
        output_path: str (optional, default: '~/brca_v01_motifism_{}.pkl')
            these can be aggregated later
        
        Returns:
        results: dict
            aggregate of results
        """
        assert os.path.exists(self.file), 'need to have a valid mut file of csv'

        peak_reader = regdiffdata.load_peak_locs(
            csv_file=self.file,
            chunksize=self.chunksize,)
        if self.chunksize is None:
            peak_reader = [peak_reader] # just one df

        if self.verbose:
            tic = time.time()
            n = 0
        for i, chunk in enumerate(peak_reader):
            if self.verbose:
                n += chunk.shape[0]
                print(f"starting chunk {i}:", chunk.shape)

            # get preproceesed chunk
            chunk = self.pp_chunk(chunk)

            if not os.path.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir)
            
            f = os.path.join(self.tmp_dir, '{}_chunk{}_{}.pkl'.format(self.file_specifier, i, 'mut' if self.mutseq_col_key is not None else 'ref'))
            chunk.to_pickle(f)
            job_name = 'ism{}{}_{}'.format(self.file_specifier, 'mut' if self.mutseq_col_key is not None else 'ref', i)
            while not regdiffdata.can_job_be_submitted():
                time.sleep(1)
            qsub_command = f"bash {self.script} {job_name} {self.file_specifier} sub {f}"
            if self.verbose:
                print('  command: $ ', qsub_command)
            os.system(qsub_command)
            if self.verbose:
                print('submitted chunk {} after {:.2f}-s'.format(i, time.time() - tic))


    
class ism_tst(object):
    def __init__(self, ):
        self.lastupdate = '230315'

        # get mutdata
        from reg_diffs.scripts import tomtom_query as tt
        mutdata = tt.mutdata().get_mutdict()
        sample = 'brca'
        mutdf = mutdata[sample.upper()]
        self.df = mutdf.sample(5, replace=False)

    def chk_get_seq(self, chk_n_flanks=[0, 5, 10], chk_with_output_length: bool = True):
        """
        TODO: 
          1. bug in Arvind's code that get_ref_alt_seq doesn't work with
             uneven output_lengths, preventing this chk of chk_with_output_length
        To run:
          open script, import this file and run: 
          >>> tst = glism.ism_tst().chk_get_seq()
        """
        assert chk_with_output_length, 'gt fx (get_ref_alt_seq) does not work with odd OLs. See TODO'
        # init ism
        chk_class = ism(file_specifier='blah')
        from seq2atac.analysis.enrichment_utils import get_refalt_sequence
        from reg_diffs.scripts import tomtom_query as tt

        # check that ref and alt align
        for n_flank in chk_n_flanks:
            fasta_seq = regdiffdata.get_fasta()
            ref_get_seq = [chk_class.get_seq(
                    fasta_seq=fasta_seq, 
                    mut_seq_at_start=None,
                    chrom=r[chk_class.chrom_key],
                    start=r[chk_class.start_key],
                    end=r[chk_class.end_key],
                    n_flank=n_flank,
                    output_length=chk_class.output_length if chk_with_output_length else None,
                ) for i, r in self.df.iterrows()]
            alt_get_seq = [chk_class.get_seq(
                    fasta_seq=fasta_seq, 
                    mut_seq_at_start=r['Tumor_Seq_Allele2'],
                    chrom=r[chk_class.chrom_key],
                    start=r[chk_class.start_key],
                    end=r[chk_class.end_key],
                    n_flank=n_flank,
                    output_length=chk_class.output_length if chk_with_output_length else None,
                ) for i, r in self.df.iterrows()] 
            
            from seq2atac.analysis import fasta_seq
            refalt_crct = get_refalt_sequence(
                df=self.df, 
                input_width=chk_class.output_length if chk_with_output_length else 1 + 2*n_flank, 
                fasta_seq=fasta_seq)
            # they're one off so for now, assert 
            for i, (ref, alt) in enumerate(zip(ref_get_seq, alt_get_seq)):
                assert ref[:-1]==refalt_crct[0][i][1:], 'ref with shift must match'
                assert alt[:-1]==refalt_crct[1][i][1:], 'alt with shift must match'
        # return last for fun
        return {'refalt_crct': refalt_crct, 'ref_get_seq': ref_get_seq, 'alt_get_seq': alt_get_seq}
    
    def intercept_chunk_pp(self):
        chk_class = ism(file_specifier='blah', mutseq_col_key=None)
        return chk_class.pp_chunk(self.df)
    
    def chk_pp_chunk(self, ):
        """
        To run:
          ```
          >>> import sys
          >>> sys.path.append('/./')
          >>> from reg_diffs.scripts import genloc_ism as glism
          >>> tst = glism.ism_tst()
          >>> tstout = tst.chk_pp_chunk()
          ```
        """

        chk_class = ism(file_specifier='blah', 
                        mutseq_col_key=None,) # for ref only

        from seq2atac.analysis import fasta_seq
        from seq2atac.analysis.enrichment_utils import get_refalt_sequence
        refalt_crct = get_refalt_sequence(
                df=self.df, 
                input_width=chk_class.output_length, 
                fasta_seq=fasta_seq)
        tstout = {'chunk_df': chk_class.pp_chunk(self.df), 'gt_seqs': refalt_crct}

        for i in range(self.df.shape[0]):
            assert tstout['gt_seqs'][0][i][1:] == tstout['chunk_df'].loc[(tstout['chunk_df']['source']=='seq'), 'seqs'].iloc[i][:-1]
        return tstout

        

class eval_glism(object):
    """
    TODO:
      - add eval functionality

    Examples:
          to run:
          ```
          import os
          import globeval
          import sys
          sys.path.append('/./')
          output_dir = './tmp/ism/results/'
          res_files = glob.glob(os.path.join(output_dir, '{}_chunk*.pkl'.format('*')))
          eval_ = glism.eval_glism(res_files=res_files).agg_res_files()
          id_cols = ['Chromosome', 'hg38_start', 'hg38_end', 'sample']
          eval_ = pd.pivot(eval_, index=id_cols, columns='seq_type', values='log2FC_seqVshuffled').reset_index()
          eval_['ref-mut'] = eval_['ref'] - eval_['mut']
          # select top 95th pctile (bottom 5th) of high-effect contexts mutations, then count tomtom_query
          ```
    """
    def __init__(self, 
                 res_files: list, 
                 chrom_key: str = 'Chromosome',
                 start_key: str = 'hg38_start',
                 end_key: str = 'hg38_end',
                 id_key: str = 'mutation_id',
                 verbose: bool = True,):
        """
        Arguments:
          res_files: list
            specify the result files to merge on. Ok if just one
        """
        self.res_files = res_files
        self.verbose = verbose
        if self.verbose:
            print('found {} files to process'.format(len(res_files)))
        self.chrom_key = chrom_key
        self.start_key = start_key
        self.end_key = end_key
        self.id_key = id_key
        
    def agg_res_files(self, ):
        """Convert list of files into one data frame assuming specifier of importance
           is last of split('_') e.g., "ref" for filename = 'brca_chunk652_ref.pkl'
        """
        df = pd.DataFrame()
        for i, f in enumerate(self.res_files):
            sample = os.path.split(f)[1].split('.pkl')[0].split('_')[0]
            seq_type = os.path.split(f)[1].split('.pkl')[0].split('_')[-1]
            dt = pd.read_pickle(f)
            dt = self.eval_chunk_df(dt)
            dt['sample'] = [sample] * dt.shape[0]
            dt['seq_type'] = [seq_type] * dt.shape[0]
            df = pd.concat([df, dt])
        self.res_df = df
        return df
            



    def eval_chunk_df(
        self,
        chunk_df: pd.DataFrame, 
        ):
        """Process results file of n_model outputs
        
        Arguments:
        include_null: bool (optional, default: True)
            Construct the null from what we have.
        method: str (optional, default: 'ave_over_folds'
            One of ['ave_over_folds'], where 'ave_over_folds' means that first,
            the model output is averaged per seq across each of the folds
            
        Returns:
        chunk_res: dict
            A dict with a null key of log2 differenes or, otherwise, the chunked_df
        
        """
        chunk_df['model_ave'] = chunk_df.loc[:, [i for i in chunk_df.columns if len(re.findall('(model)(.*)(out)', i)) > 0]].mean(1)
        
        # main effect
        chunk_reduced = chunk_df.groupby([self.chrom_key, self.start_key, self.end_key, self.id_key, 'source']).mean().reset_index()
        # # recalc aves to first reduce over shuffled seqs
        # chunk_reduced['model_ave'] = chunk_reduced.loc[:, [i for i in chunk_reduced.columns if len(re.findall('(model)(.*)(out)', i)) > 0]].mean(1)
        
        ## calc diffsn 
        chunk_diff = chunk_reduced.loc[
            chunk_reduced['source']=='seq', 
            [self.chrom_key, self.start_key, self.end_key, self.id_key, 'model_ave']].merge(
            chunk_reduced.loc[
                chunk_reduced['source']=='shuffled_seqs', 
                [self.chrom_key, self.start_key, self.end_key, self.id_key, 'model_ave']],
            on=[self.chrom_key, self.start_key, self.end_key, self.id_key,], 
            suffixes=['_seq', '_shuffled'])
        chunk_diff['log2FC_seqVshuffled'] = np.log2(chunk_diff['model_ave_seq']) - np.log2(chunk_diff['model_ave_shuffled'])
        
        # null distribution
        if False:
            null_dist = []
            dt = chunk_df.loc[chunk_df['source']=='shuffled_seqs'].groupby([self.chrom_key, self.start_key, self.end_key])
            # n = dt.size().shape[0]
            for i, (idx, grp) in enumerate(dt):
                a = grp['model_ave'].to_numpy()
                null_dist.append(np.mean([np.log2(a[i]) - np.log2(a[j]) for i, j in list(itertools.combinations(list(range(a.shape[0])), 2))]))

            return chunk_diff, null_dist
        else:
            return chunk_diff


     

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--file_specifier', type=str, 
        help="s for sample... specify file or cancer to use, e.g., brca or null")
    parser.add_argument('-f', '--file', default=None, type=str, 
        help="temporary filename of the chunk dataframe with chrom start end cols")
    parser.add_argument('-t', '--program_switch', default='main', type=str,
        help="whether to run the main script which calls a series of jobs or within that job, whether the sub analyses are called")
    args = parser.parse_args()
    print('\nArgs specified:')
    print(' ', args)

    #########################################################################
    # specify IO directories
    #########################################################################
    tmp_dir = './tmp/ism/data/'
    output_dir = './tmp/ism/results/gnomadv2/'
    #########################################################################
    
    if args.program_switch == 'main':
        # create files 
        if 'genomewide' in args.file_specifier.lower() or 'gwide' in args.file_specifier.lower():
            args.file_specifier = args.file_specifier.split('_')[0]
            # HERE
            file = './tmp/TCGA/mutation_prioritization/somatic_filtered/{}_filtered_somatic_annotated.pkl'.format(args.file_specifier.upper())
            # convert to csv to allow for chunking
            new_file = os.path.join(tmp_dir, os.path.splitext(os.path.split(file)[1])[0])
            df = pd.read_pickle(file)
            # chk for non-canonical seqs
            df = ism.filter_nonACGT(df, n_flank=5, )
            df.to_csv(new_file)
            file = new_file

        elif 'gnmdv2' in args.file_specifier.lower():
            args.file_specifier = args.file_specifier.split('_')[0]
            file = './tmp/TCGA/mutation_prioritization/matching_experiments_scored/closest_pancan_match/{}_gnomad_matched.pkl'.format(args.file_specifier.upper())
            # convert to csv to allow for chunking
            new_file = os.path.join(tmp_dir, os.path.splitext(os.path.split(file)[1])[0])
            df = pd.read_pickle(file)
            # chk for non-canonical seqs
            df = ism.filter_nonACGT(df, n_flank=5, )
            df.to_csv(new_file)
            file = new_file

        elif 'gnomad' in args.file_specifier.lower():
            args.file_specifier = args.file_specifier.split('_')[0]
            file = './tmp/TCGA/mutation_prioritization/matching_experiments_scored/abstain_ap1_ctcf_v7/{}_gnomad_matched.pkl'.format(args.file_specifier.upper())
            # convert to csv to allow for chunking
            new_file = os.path.join(tmp_dir, os.path.splitext(os.path.split(file)[1])[0])
            df = pd.read_pickle(file)
            # chk for non-canonical seqs
            df = ism.filter_nonACGT(df, n_flank=5, )
            df.to_csv(new_file)
            file = new_file


        else:
            file = os.path.join(tmp_dir, args.file_specifier.lower().strip() + '_gofmut.csv')
            if not os.path.exists(file):
                from reg_diffs.scripts import tomtom_query as tt
                mutdf = tt.mutdata().get_mutdf(sample=args.file_specifier.lower(), subset='gof')
                # mutdf = ism.filter_nonACGT(mutdf, n_flank=5, )
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                # mutdf = filterout_noncanonical_seq(mutdf)
                mutdf.to_csv(file)
                print('mutdf written to: ', file)

        ref_run = ism(
            file=file, 
            file_specifier=args.file_specifier,
            tmp_dir=tmp_dir, 
            mutseq_col_key=None, 
            chunksize=1e6, 
            n_shuffles=6, 
            n_flank=5, 
            output_length=1364, 
            chrom_key='Chromosome', 
            start_key='hg38_start', 
            end_key='hg38_end', 
            script='/./reg_diffs/experiments/glism.sh', 
            verbose=True
        )

        alt_run = ism(
            file=file,
            file_specifier=args.file_specifier,
            tmp_dir=tmp_dir, 
            mutseq_col_key='Tumor_Seq_Allele2', 
            chunksize=1e6, 
            n_shuffles=6, 
            n_flank=5, 
            output_length=1364, 
            chrom_key='Chromosome', 
            start_key='hg38_start', 
            end_key='hg38_end', 
            script='/./reg_diffs/experiments/glism.sh', 
            verbose=True,
        )
        
        # actually run the pipelines
        ref_run.run_ism_pipeline()
        alt_run.run_ism_pipeline()

        print('\Submitted both ref and alt jobs.')
        print('DONE.')

    elif args.program_switch == 'sub':

        # specify paths to weights
        path_to_weights = './tmp/TCGA/models_250_1364_minibatch_prejitter/' + args.file_specifier.upper() + '/fold_{}/model.h5'
        assert os.path.exists(path_to_weights.format(0)), 'invalid model weights path specified'
        
        print('Starting file: {}'.format(args.file))
        chunk = ism.get_model_output(
            chunk_file=args.file,
            path_to_weights=path_to_weights,
            batch_size=128,
            n_folds=5,
            output_dir=output_dir,
            clean_mem=True,
            store_seqs=False,
            verbose=True,
        )

    elif args.program_switch == 'eval':
        # call file_specifier arg as all
        res_files = glob.glob(os.path.join(output_dir, '{}_chunk*.pkl'.format('*' if args.file_specifier=='all' else args.file_specifier)))
        eval_glism(res_files=res_files)





    
    
    