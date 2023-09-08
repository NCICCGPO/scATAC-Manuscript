import glob
import os
import pandas as pd
import numpy as np
import pickle
import time
import re
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

def proc_chunkdf(
    chunk_df: pd.DataFrame, 
    include_null: bool = True,
    name_key: str = 'group_name',
    chrom_key: str = 'seqnames',
    start_key: str = 'start',
    end_key: str = 'end',
    method: str = 'ave_over_folds'):
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
    if method == 'ave_over_folds':
        chunk_df['model_ave'] = chunk_df.loc[:, [i for i in chunk_df.columns if len(re.findall('(model)(.*)(out)', i)) > 0]].mean(1)
        
        chunk_res = {}
        # main effect
        chunk_reduced = chunk_df.groupby([name_key, chrom_key, start_key, end_key, 'source']).mean().reset_index()
        # # recalc aves to first reduce over shuffled seqs
        # chunk_reduced['model_ave'] = chunk_reduced.loc[:, [i for i in chunk_reduced.columns if len(re.findall('(model)(.*)(out)', i)) > 0]].mean(1)
        
        ## calc diffsn 
        chunk_diff = chunk_reduced.loc[
            chunk_reduced['source']=='seq', 
            [name_key, chrom_key, start_key, end_key, 'model_ave']].merge(
            chunk_reduced.loc[
                chunk_reduced['source']=='shuffled_seqs', 
                [name_key, chrom_key, start_key, end_key, 'model_ave']],
            on=[name_key, chrom_key, start_key, end_key,], 
            suffixes=['_seq', '_shuffled'])
        chunk_res['chunk_diff'] = chunk_diff
        
        # null distribution
        if include_null:
            chunk_res['null_dist_log2diff'] = []
            dt = chunk_df.loc[chunk_df['source']=='shuffled_seqs'].groupby([name_key, chrom_key, start_key, end_key])
            # n = dt.size().shape[0]
            for i, (idx, grp) in enumerate(dt):
                a = grp['model_ave'].to_numpy()
                chunk_res['null_dist_log2diff'].append(np.mean([np.log2(a[i]) - np.log2(a[j]) for i, j in list(itertools.combinations(list(range(a.shape[0])), 2))]))
    else:
        raise NotImplementedError 


        
    return chunk_res
    
def proc_chunkfiles(
    file_list: list or str,
    out_file: str or None = None, 
    include_null: bool = True,
    name_key: str = 'group_name',
    chrom_key: str = 'seqnames',
    start_key: str = 'start',
    end_key: str = 'end',
    method: str = 'ave_over_folds',
    clean_mode: bool = False):
    """
    Arguments:
      clean_mode: bool (optional, default: False)
        del file with stored model outputs after aggregating diffs?
    """
    
    chunk_res_agg = {}
    
    if isinstance(file_list, str):
        # assume it's a filepath
        with open(file_list, 'rb') as f:
            file_list = pickle.load(f)
            f.close()
        if clean_mode:
            # delete file list and original file
            os.remove(file_list)
    
    for i, f in enumerate(file_list):
        chunk = pd.read_pickle(f)
        
        chunk_res_agg[os.path.split(f)[1].split('.pkl')[0]] = proc_chunkdf(
            chunk_df=chunk, 
            include_null=include_null,
            name_key=name_key,
            chrom_key=chrom_key,
            start_key=start_key,
            end_key=end_key,
            method=method)
        
        if clean_mode:
            os.remove(f)
            
    if out_file is not None:
        with open(out_file, 'wb') as f:
            pickle.dump(chunk_res_agg, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
    else:
        return chunk_res_agg

def main(
    filepath: str = '',
    n_files_per_job: int = 10,
    out_filepath: str = './tmp/brca_v01/diff_results/',
    script: str = '/./reg_diffs/scripts/eval_ism.sh'):
    """
    Arguments:
    """
    if out_filepath is not None and not os.path.exists(out_filepath):
        os.makedirs(out_filepath)

    files = glob.glob(os.path.join(filepath, '*.pkl'))
    
    # make temporary dir
    proc_tmp_dir = './reg_diffs_tmp/' # tmp_dir for processing only
    if not os.path.exists(proc_tmp_dir):
        os.makedirs(proc_tmp_dir)
    for i in range(0, len(files), n_files_per_job):
        file_grp = files[i:i+n_files_per_job]
        # write to tmp file and send off to sub jobs
        tmp_proc_file = os.path.join(proc_tmp_dir, 'chunk_grp{}.pkl'.format(i))
        with open(tmp_proc_file, 'wb') as f:
            pickle.dump(file_grp, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        # fire off job
        job_name = 'eval{}chunkset'.format(i)
        qsub_command = f"bash {script} {job_name} sub {tmp_proc_file} {out_filepath}"
        os.system(qsub_command)
        
    
    # # remove tmp dir
    # time.sleep(5*60) # error prone, since may have to wait for jobs
    # os.rmdir(proc_tmp_dir)
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='specify write out name')
    parser.add_argument('-fp', '--filepath', type=str,
                        help="filepath to .pkl result files to process")
    parser.add_argument('-ps', '--program_switch', type=str,
                        help='specify whether to kick off jobs through main or sub')
    parser.add_argument('-ofp', '--out_filepath', 
                        default='./tmp/brca_v01/diff_results/')
    args = parser.parse_args()
    
    ################ # modify ################
    # brca results: './tmp/reg_diffs/results/tmp/brca_v01/'
    n_files_per_job = 20
    clean_mode = True
    ##########################################
    
    if args.program_switch == 'main':
        main(
            filepath=args.filepath,
            n_files_per_job=n_files_per_job,
            out_filepath=args.out_filepath,
            script='/./reg_diffs/experiments/eval_ism.sh')
    else:
        results_outfile = os.path.join(args.out_filepath, args.name + '.pkl')
        proc_chunkfiles(
            file_list=args.filepath,
            out_file=results_outfile, 
            include_null=True,
            name_key='group_name',
            chrom_key='seqnames',
            start_key='start',
            end_key='end',
            method='ave_over_folds',
            clean_mode=clean_mode)
        
        

    
    