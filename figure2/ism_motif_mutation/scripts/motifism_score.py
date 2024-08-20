import pandas as pd

import regdiffmodels ## cwd
import regdiffdata ## cwd

import re
import os
import numpy as np
import gc
import pandas as pd
import time
import tensorflow as tf
import glob
import datetime

def create_chunks(
    motif_file: str,
    chunksize: int or None,
    n_shuffles: int = 6,
    window_size: int = 20,
    output_length: int = 1364,
    name_key: str = 'group_name',
    chrom_key: str = 'seqnames',
    start_key: str = 'start',
    end_key: str = 'end',
    verbose: bool = False,
    tmp_dir: str = './temporary_files_ism/brca/', 
    ):

    assert window_size % 2 == 0, 'pick symmetrical window size. '
    peak_reader = regdiffdata.load_peak_locs(
        csv_file=motif_file,
        chunksize=chunksize,)
    if chunksize is None:
        peak_reader = [peak_reader] # just one df

    fasta_seq = regdiffdata.get_fasta()
    L = (output_length // 2) - (window_size // 2)
    R = (output_length // 2) + (window_size // 2) + 1

    if verbose:
        tic = time.time()
        n = 0
    for i, chunk in enumerate(peak_reader):
        if verbose:
            n += chunk.shape[0]
        chunk['seq'] = [regdiffdata.get_refseq_window(fasta_seq, r[chrom_key], r[start_key], r[end_key]) for i, r in chunk.iterrows()]
        chunk['shuffled_seqs'] = chunk['seq'].apply(lambda x: np.hstack([x[:L] + regdiffdata.dinuclShuffle(x[L:R]) + x[R:] for i in range(n_shuffles)]))
        
        # flatten chunk
        chunk = pd.melt(chunk, id_vars=[name_key, chrom_key, start_key, end_key], var_name='source', value_name='seqs')
        chunk = chunk.merge(
            chunk['seqs'].apply(pd.Series).stack().rename('seqs').reset_index(),
            right_on='level_0', left_index=True, suffixes=(['_old', '']))
        chunk = chunk[[name_key, chrom_key, start_key, end_key, 'source', 'seqs']]

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        
        chunk.to_pickle(os.path.join(tmp_dir, 'chunk{}.pkl'.format(i)))
    if verbose:
      print('done chunking')
    return None
      
def get_model_output(
    chunk_file: str,
    path_to_weights: str,
    batch_size: int = 128,
    n_folds: int = 5,
    output_path: str = './ism_cleaning_output/brca/',
    verbose: bool = False,
    clean_mem: bool = True,
    store_seqs: bool = False,
    ):

    chunk = pd.read_pickle(chunk_file)
    # now that it is in memory, remove
    if verbose:
        tic = time.time()
        n = chunk.shape[0]
    # iterate through each model, then append width
    if '{}' not in path_to_weights:
        model_state_files = [path_to_weights]
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
    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        tmp_name = os.path.split(chunk_file)[1].split('.pkl')[0]
        # clean up 
        if not store_seqs:
            del chunk['seqs'] 
        chunk.to_pickle(os.path.join(output_path, tmp_name + '.pkl'))
    if verbose:
        print('through chunk {} in {}-s ({} motifs)'.format(tmp_name, time.time() - tic, n))
    if clean_mem:
        os.remove(chunk_file)
    return chunk

def main(
    motif_file: str,
    chunksize: int or None,
    path_to_weights: str,
    n_shuffles: int = 6,
    window_size: int = 20,
    output_length: int = 1364,
    batch_size: int = 128,
    n_folds: int = 5,
    name_key: str = 'group_name',
    chrom_key: str = 'seqnames',
    start_key: str = 'start',
    end_key: str = 'end',
    tmp_dir: str = './temporary_files_ism/brca/',
    output_path: str = './ism_cleaning_output/brca/',
    verbose: bool = False,
    script: str = './motifism_launch.sh',
    file_specifier: str = 'brca',
    ):
    assert window_size % 2 == 0, 'pick symmetrical window size. '
    peak_reader = regdiffdata.load_peak_locs(
        csv_file=motif_file,
        chunksize=chunksize,)
    if chunksize is None:
        peak_reader = [peak_reader] # just one df

    fasta_seq = regdiffdata.get_fasta()
    L = (output_length // 2) - (window_size // 2)
    R = (output_length // 2) + (window_size // 2) + 1

    if verbose:
        tic = time.time()
        n = 0
    for i, chunk in enumerate(peak_reader):
        if verbose:
            n += chunk.shape[0]
        # 
        chunk = chunk.loc[:, [name_key, chrom_key, start_key, end_key]] # reduce size and remove chance for addl columns
        chunk['seq'] = [regdiffdata.get_refseq_window(fasta_seq, r[chrom_key], r[start_key], r[end_key], output_length=output_length) for i, r in chunk.iterrows()]
        chunk['shuffled_seqs'] = chunk['seq'].apply(lambda x: np.hstack([x[:L] + regdiffdata.dinuclShuffle(x[L:R]) + x[R:] for i in range(n_shuffles)]))
        
        # flatten chunk
        chunk = pd.melt(chunk, id_vars=[name_key, chrom_key, start_key, end_key], var_name='source', value_name='seqs')
        chunk = chunk.merge(
            chunk['seqs'].apply(pd.Series).stack().rename('seqs').reset_index(),
            right_on='level_0', left_index=True, suffixes=(['_old', '']))
        chunk = chunk[[name_key, chrom_key, start_key, end_key, 'source', 'seqs']]

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        
        f = os.path.join(tmp_dir, 'chunk{}.pkl'.format(i))
        chunk.to_pickle(f)
        job_name = 'ism{}_{}'.format(file_specifier, i)
        while not regdiffdata.can_job_be_submitted():
            time.sleep(1)
        qsub_command = f"bash {script} {job_name} {file_specifier} sub {f}"
        os.system(qsub_command)
        if verbose:
            print('submitted chunk {} after {:.2f}-s'.format(i, time.time() - tic))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--file_specifier', type=str, 
        help="s for sample... specify file or cancer to use, e.g., brca or brca11")
    parser.add_argument('-f', '--file', default=None, type=str, 
        help="temporary filename of the chunk dataframe with chrom start end cols")
    parser.add_argument('-t', '--program_switch', default='main', type=str,
        help="whether to run the main script which calls a series of jobs or within that job, whether the sub analyses are called")
    parser.add_argument('-cs', '--chunk_size', type=int, default=1e5)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    args = parser.parse_args()
    print('\nArgs specified:')
    print(' ', args)

    path_to_weights = './models/' + args.file_specifier.upper() + '/fold_{}/model.h5' ### TODO: point to models
    motifs = './uncleaned_motifs/vierstra_Archetype_{}agg.csv.bed'.format(args.file_specifier.upper()) ### TODO: point to unclean motifs
    print('path_to_weights', path_to_weights)
    print('motifs:', motifs)
    
    # argument for output
    tmp_dir = './temporary_files_ism/{}/'.format(args.file_specifier.lower())
    output_path = './ism_cleaning_output/{}/'.format(args.file_specifier.lower())
    if not os.path.exists(tmp_dir):
      os.makedirs(tmp_dir)
    if not os.path.exists(output_path):
      os.makedirs(output_path)

    ################ # #### ################

    if args.program_switch == 'main':
        main(
            motif_file=motifs,
            chunksize=args.chunk_size,
            path_to_weights=path_to_weights,
            n_shuffles=6,
            window_size=20,
            output_length=1364,
            batch_size=args.batch_size,
            n_folds=5,
            name_key='group_name',
            chrom_key='seqnames',
            start_key='start',
            end_key='end',
            tmp_dir=tmp_dir,
            output_path=output_path,
            verbose=True,
            script='./motifism_launch.sh',
            file_specifier=args.file_specifier,
            )
    else:
        print('Starting file: {}'.format(args.file))
        chunk = get_model_output(
            chunk_file=args.file,
            path_to_weights=path_to_weights,
            batch_size=args.batch_size,
            n_folds=5,
            output_path=output_path,
            verbose=True,
            clean_mem=True,
        ) 
