import pandas as pd

import regdiffmodels
import regdiffdata

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
        tmp_dir: str = './temporary_files_mut_ism/',
        mutseq_col_key: str or None = 'Tumor_Seq_Allele2', # if none, ref seq used
        chunksize: int or None = None,
        n_shuffles: int = 6,
        n_flank: int = 5, 
        output_length: int = 1364,
        chrom_key: str = 'Chromosome',
        start_key: str = 'hg38_start',
        end_key: str = 'hg38_end',
        id_key: str = 'mutation_id',
        script: str = './mutationism_launch.sh',
        verbose: bool = True,
    ):
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
            print('WARNING. Poor file specification saved')
        if verbose:
            print('through chunk {} in {}-s ({} motifs)'.format(tmp_name, time.time() - tic, n))
        if clean_mem and not isinstance(chunk_file, pd.DataFrame):
            os.remove(chunk_file)
        return chunk

    
    def run_ism_pipeline(self,):
        """in silico mutagenesis for file, in chunks, specified by name and location
        
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



class eval_glism(object):

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
    tmp_dir = './temporary_files_mut_ism/'
    output_dir = './mut_ism_cleaning_output/'
    #########################################################################
    
    if args.program_switch == 'main':

        args.file_specifier = args.file_specifier.split('_')[0]
        file = args.file

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
            script='./mutationism_launch.sh', 
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
            script='./mutationism_launch.sh', 
            verbose=True,
        )
        
        # actually run the pipelines
        ref_run.run_ism_pipeline()
        alt_run.run_ism_pipeline()

        print('\Submitted both ref and alt jobs.')
        print('DONE.')

    elif args.program_switch == 'sub':

        # specify paths to weights
        path_to_weights = './models/' + args.file_specifier.upper() + '/fold_{}/model.h5'
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
    