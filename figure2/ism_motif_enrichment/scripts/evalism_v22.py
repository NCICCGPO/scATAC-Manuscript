"""

Examples:
  $ bash /./reg_diffs/experiments/evalism_v22.sh GBM45_cloneB
  - see ~/reg_diffs/experiments/evalism_v22.sh

"""

import glob
import os
import pandas as pd
import numpy as np
import pickle
import time
import re
import itertools
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency

class eval_ism():
    def __init__(
        self,
        filepath: list or str,
        outfile: str or None = None,
        name_key: str = 'group_name',
        chrom_key: str = 'seqnames',
        start_key: str = 'start',
        end_key: str = 'end',
        method: str = 'ave_over_folds', 
        clean_mode: bool = False,
        debug_mode: bool = False,
    ):
        self.filepath = filepath
        self.outfile = outfile
        self.name_key = name_key
        self.chrom_key = chrom_key
        self.start_key = start_key
        self.end_key = end_key
        self.method = method
        self.clean_mode = clean_mode
        self.debug_mode = debug_mode


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
        if self.method == 'ave_over_folds':
            chunk_df['model_ave'] = chunk_df.loc[:, [i for i in chunk_df.columns if len(re.findall('(model)(.*)(out)', i)) > 0]].mean(1)
            
            # main effect
            chunk_reduced = chunk_df.groupby([self.name_key, self.chrom_key, self.start_key, self.end_key, 'source']).mean().reset_index()
            # # recalc aves to first reduce over shuffled seqs
            # chunk_reduced['model_ave'] = chunk_reduced.loc[:, [i for i in chunk_reduced.columns if len(re.findall('(model)(.*)(out)', i)) > 0]].mean(1)
            
            ## calc diffsn 
            chunk_diff = chunk_reduced.loc[
                chunk_reduced['source']=='seq', 
                [self.name_key, self.chrom_key, self.start_key, self.end_key, 'model_ave']].merge(
                chunk_reduced.loc[
                    chunk_reduced['source']=='shuffled_seqs', 
                    [self.name_key, self.chrom_key, self.start_key, self.end_key, 'model_ave']],
                on=[self.name_key, self.chrom_key, self.start_key, self.end_key,], 
                suffixes=['_seq', '_shuffled'])
            
            # null distribution
            if True:
                null_dist = []
                dt = chunk_df.loc[chunk_df['source']=='shuffled_seqs'].groupby([self.name_key, self.chrom_key, self.start_key, self.end_key])
                # n = dt.size().shape[0]
                for i, (idx, grp) in enumerate(dt):
                    a = grp['model_ave'].to_numpy()
                    null_dist.append(np.mean([np.log2(a[i]) - np.log2(a[j]) for i, j in list(itertools.combinations(list(range(a.shape[0])), 2))]))
        else:
            raise NotImplementedError 

        return chunk_diff, null_dist
    
    def proc_motifism_file(
        self,
        ):
        """
        Arguments:
        clean_mode: bool (optional, default: False)
            del file with stored model outputs after aggregating diffs?
        """
        if self.debug_mode:
            print(self.outfile)
        
        chunk_res_agg = pd.DataFrame()
        null_dists = []

        print('Checking for chunks at:', os.path.join(self.filepath, 'chunk*.pkl'))

        chunk_list = glob.glob(os.path.join(self.filepath, 'chunk*.pkl'))
        print('found {} files. Starting processing...'.format(len(chunk_list)))
        
        for i, f in enumerate(chunk_list):
            chunk_name = os.path.split(f)[1].split('.pkl')[0]
            chunk = pd.read_pickle(f)
            df, null = self.eval_chunk_df(chunk)
            chunk_res_agg = pd.concat([chunk_res_agg, df]) # ignore_index = True?
            null_dists += null
            
            if self.clean_mode and False: # failsafe!!! Don't delete these files until someone complains about mem
                os.remove(f)

            print('...  done with:', f)
                
        if self.outfile is not None:
            with open(self.outfile, 'wb') as f:
                pickle.dump({'df': chunk_res_agg, 'null_dists': null_dists}, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
        return chunk_res_agg, null_dists

def main(
    filepath: str,
    outfile: str or None = None,
    ):
    """
    Arguments:
    """
    if outfile is not None and not os.path.exists(os.path.split(outfile)[0]):
        os.makedirs(os.path.split(outfile)[0])

    evaluator = eval_ism(filepath=filepath, outfile=outfile)
    return evaluator.proc_motifism_file()


# add to class but allow to be imported? 
def get_evals(
    sample: str = 'brca10',
    outfile_pattern: str = './tmp/results/{}_v22.pkl',
):
    eval_dict = pd.read_pickle(outfile_pattern.format(sample))
    print('\n  ism file: {}'.format(outfile_pattern.format(sample)))
    return eval_dict['df'], eval_dict['null_dists']

def clean_from_df_null(
    df: pd.DataFrame,
    null_dist: list or np.ndarray,
    name_key: str = 'group_name',
    apply_filter: bool = False,
    return_thresh: bool = True
):
    df['es'] = np.log2(df['model_ave_seq']) - np.log2(df['model_ave_shuffled'])
    df = df.merge(df.groupby(name_key).count()['es'], how='left', left_on=name_key, right_index=True, suffixes=['', '_cnt_before'])
    
    
    # add key, apply filter
    df['enrichment'] = 'not cleaned'
    thresh = np.quantile(null_dist, 0.95)
    df.loc[df['es'] > thresh, 'enrichment'] = 'cleaned'

    if apply_filter:
        # 2. filter
        df = df.loc[df['enrichment']=='cleaned', :]

    # 3. count motifs, compute ratio
    df = df.merge(df.loc[df['enrichment']=='cleaned', :].groupby(name_key).count()['es'], how='left', left_on=name_key, right_index=True, suffixes=['', '_cnt_after'])
    # df['short_name'] = df['names'].astype(str).apply(lambda x: x.split('|')[-1])
    df['cnt_after/before'] = df['es_cnt_after'] / df['es_cnt_before']
    
    if return_thresh:
        return df, thresh
    else:
        return df

def cleaning(
    sample: str = 'blca', 
    save: bool = False,
    apply_filter: bool = True,
    name_key: str = 'group_name',
    outfile_pattern: str = './tmp/results/{}_v22.pkl',
):
    df, null_dist = get_evals(sample=sample, outfile_pattern=outfile_pattern)
    df['es'] = np.log2(df['model_ave_seq']) - np.log2(df['model_ave_shuffled'])
    df = df.merge(df.groupby(name_key).count()['es'], how='left', left_on=name_key, right_index=True, suffixes=['', '_cnt_before'])
    
    
    # add key, apply filter
    df['enrichment'] = 'not cleaned'
    df.loc[df['es'] > np.quantile(null_dist, 0.95), 'enrichment'] = 'cleaned'

    if apply_filter:
        # 2. filter
        df = df.loc[df['enrichment']=='cleaned', :]

    # 3. count motifs, compute ratio
    df = df.merge(df.loc[df['enrichment']=='cleaned', :].groupby(name_key).count()['es'], how='left', left_on=name_key, right_index=True, suffixes=['', '_cnt_after'])
    # df['short_name'] = df['names'].astype(str).apply(lambda x: x.split('|')[-1])
    df['cnt_after/before'] = df['es_cnt_after'] / df['es_cnt_before']
    
    if save:
        df.to_csv('./tmp/results/{}_motifismeval_cleaned.csv'.format(sample))
    
    return df

def merge_dfdict(
    data: dict,
    sub_cols = ['names', 'es_cnt_before', 'es_cnt_after'],
    name_key: str = 'names',
    ):
    sample_names = []
    for i, (k, v) in enumerate(data.items()):
        sample_names.append(k)
        # rank
        v = v.sort_values(by='es', ascending=False)
        v = v.drop_duplicates(subset=name_key)
        if sub_cols is not None:
            # reduce the size of the dfs
            v = v.loc[:, sub_cols]
        if i == 0:
            df = v
        else:
            df = df.merge(v, on=name_key, how='outer', suffixes=['_{}'.format(sample_names[i-1]), ''])
    df = df.rename(columns={kk:'{}_{}'.format(kk, k) for kk in sub_cols[1:]})
    df = df.fillna(0) # count as 0 those motif absences
    df = df.set_index(name_key)
    return df

def chi2ovr(
    data: dict,
    groups: dict or None = None,
    notB_key: str = 'es_cnt_before', 
    B_key: str = 'es_cnt_after',
    return_df: bool = False,
    verbose: bool = False,
    return_contingency_tab: bool = False,
    name_key: str = 'names',
    ): 
    """
    Arguments:
      data: dict 
        dictionary of pd.DataFrames with sample_names as keys
    """
    
    if verbose: 
        tic = time.time()
    df = merge_dfdict(data, sub_cols=[name_key, notB_key, B_key], name_key=name_key)
    if verbose:
        print('data merged into one df in {:.0f}-s'.format(time.time() - tic))
        
    out = {}
    count = 0
    for k in data.keys():
        if groups is not None:
            group = [g for g, v in groups.items() if k in v][0]
            background_keys = [kk for k, v in groups.items() for kk in v if kk not in groups[group]]
        else:
            background_keys = set(data.keys()) - set([k])
        dt = df.loc[:, ['{}_{}'.format(c, k) for c in [B_key, notB_key]]]
        dt['notA_B'] = df.loc[:, ['{}_{}'.format(B_key, ss) for ss in background_keys]].sum(1)
        dt['notA_notB'] = df.loc[:, ['{}_{}'.format(notB_key, ss) for ss in background_keys]].sum(1)
        dt = dt.loc[(dt['{}_{}'.format(B_key, k)]!=0) |
               (dt['{}_{}'.format(notB_key, k)]!=0), :]
        # after filter, add to count
        count += dt.shape[0]
        dt['OR'] = dt.apply(lambda x: 
                            ((x['{}_{}'.format(B_key, k)] / x['{}_{}'.format(notB_key, k)]) / 
                             (x['notA_B'] / x['notA_notB'])), axis=1)
        dt['p_chi2'] = dt.apply(lambda x: chi2_contingency(
            np.array([
                [x['{}_{}'.format(B_key, k)], x['{}_{}'.format(notB_key, k)]],
                [x['notA_B'], x['notA_notB']],
            ], dtype=np.int64))[1], axis=1)
        if return_contingency_tab:
            dt['tab'] = dt.apply(lambda x: np.array([
                [x['{}_{}'.format(B_key, k)], x['{}_{}'.format(notB_key, k)]],
                [x['notA_B'], x['notA_notB']],
            ], dtype=np.int64), axis=1)
        coloi = ['p_chi2', 'OR'] if not return_contingency_tab else ['p_chi2', 'OR', 'tab']
        out[k] = dt.loc[:, coloi]
        if verbose:
            print('through {}. time elapsed: {:.0f}-s'.format(k, time.time() - tic))
    print('n_tst: {}'.format(count))
    
    if return_df:
        for i, k in enumerate(out.keys()):
            dtt = pd.DataFrame(out[k])
            dtt['sample'] = k
            dtt = dtt.reset_index()
            if i==0:
                df = dtt
            else:
                df = pd.concat([df, dtt])
        df['p_bonferonni'] = df['p_chi2'] * df.shape[0]
        return df
    else:
        return out


def create_oneVrest_df(
    sample_precursor: str = 'brca',
    summarize: bool = False,
    ):
    sample_names = ['{}{}'.format(sample_precursor, i) for i in range(10, 26)]
    data = {s:cleaning(sample=s, save=False) for s in sample_names}
    groups = {'luminal': ['brca10', 'brca12', 'brca15', 'brca20', 'brca22', ],
          'basal': ['brca14', 'brca16', 'brca23', 'brca24', 'brca25', ],
          # 'her2': ['brcaconsens11', 'brcaconsens13', 'brcaconsens17', 'brcaconsens18', 'brcaconsens19', 'brcaconsens21', ],
         }
    her2_samples = ['brca11', 'brca13', 'brca17', 'brca18', 'brca19', 'brca21', ]
    df = chi2ovr({k:v for k,v in data.items() if k not in her2_samples}, verbose=True, groups=groups, return_df=True)

    # modify df 
    df.loc[df['p_chi2']==0., 'p_chi2'] = df.loc[df['p_chi2']!=0, 'p_chi2'].min()
    df['p_adj'] = df.shape[0] * df['p_chi2']
    df['nlog10_padj'] = -np.log10(df['p_adj'])

    if summarize:
        sample_names = groups['basal'] + groups['luminal']
        for i, s in enumerate(sample_names):
            if i == 0:
                dt = df.loc[(df['sample']==s) & ([True if 'C2H2_' not in i else False for i in df['names']]) & (df['OR'] > 1.) & (df['p_adj'] < 0.05), ['names', 'nlog10_padj']]
            else:
                dt = dt.merge(df.loc[(df['sample']==s) & ([True if 'C2H2_' not in i else False for i in df['names']]) & (df['OR'] > 1.) & (df['p_adj'] < 0.05), ['names', 'nlog10_padj']], on='names', how='outer', suffixes=[i-1, i])
        dt = dt.fillna(0)
        # dt = dt.rename(columns={c:sample_names[i] for i, c in enumerate([cc for cc in dt.columns if 'names' not in cc])})
        dt = dt.set_index('names')
        dt = dt.rename(columns={c:sample_names[i] for i, c in enumerate(dt.columns)})
        dt
        return df, dt
    else:
        return df
    
def agg_shap_ism_samples(
    sample_names: list = ['brca', 'blca', 'luad', 'kirp', 'kirc', 'skcm', 'gbm', 'coad', ], 
    file_pattern: str = './tmp/TCGA/peak_shap/{}/motifs_cleaned.csv',
    out_pattern: str or None = './tmp/tcga/motif_analysis/union_shap_ism_{}.csv',
    ):
    if out_pattern is not None and not os.path.exists(os.path.split(out_pattern)[0]):
            os.makedirs(os.path.split(out_pattern)[0])
    out = {}
    tic = time.time()
    for s in sample_names:
        shap_df = pd.read_csv(file_pattern.format(s.upper()), index_col=0)
        ism_df = cleaning(s)
        ism_coi = ['group_name', 'seqnames', 'start', 'end', 'es']
        shap_coi = ['group_name', 'seqnames', 'start', 'end', 'contribution']
        dt = ism_df[ism_coi].merge(shap_df[shap_coi], on=['group_name', 'seqnames', 'start', 'end'], how='outer')
        if out_pattern is not None:
            dt.to_csv(out_pattern.format(s))
        out[s] = dt
        print('  through {}\telapsed: {:.1f}'.format(s, time.time() - tic))
    return out

# change merging fx
def merge_shap_ism(
    sample: str = 'brca',
    file_pattern: str = './tmp/TCGA/peak_shap/{}',
    ismoutfile_pattern: str = './tmp/results/{}_v122.pkl',
    out_pattern: str or None = './tmp/tcga/motif_analysis/union_ismshap_{}_v122.csv',
    name_key: str = 'group_name',
    modify_names: dict or None = None,
    return_only_cleaned: bool = True,
    verbose: bool = True,
    modify_sample_name: bool = True,
):
    """Identify significant motifs in a sample
    
    Arguments: 
      modif_names: dict or None (optional, default: None)
        Modify the motif name with broader name sets by providing a dict
        key that will alter tha name_key to a new column `modified_names`, 
        then use that column to count things
    """
    if verbose:
        tic = time.time()
        print('\nMerging SHAP/ISM for {}'.format(sample))

    sample = sample.upper() if modify_sample_name else sample
        
    shap_cleaned = os.path.join(file_pattern.format(sample), 'motifs_cleaned.csv')
    shap_all = os.path.join(file_pattern.format(sample), 'motifs_with_contribution_scores.csv')
    assert os.path.exists(file_pattern.format(sample)), 'where are the shap peaks?'
    assert os.path.exists(shap_cleaned), 'need cleaned motifs'
    assert os.path.exists(shap_all), 'need uncleaned motifs'
    
    print('\n  shap file: {}'.format(shap_all))
    shap_cleaned = pd.read_csv(shap_cleaned, index_col=0)
    shap_all = pd.read_csv(shap_all, index_col=0)
    
    shap_cleaned['enrichment_shap'] = True
    shap_df = shap_all.merge(shap_cleaned, on=['group_name', 'seqnames', 'start', 'end'], how='left', suffixes=['', '_cleaned'])
    shap_df['enrichment_shap'].fillna(False, inplace=True)
    del shap_cleaned, shap_all
    
    if modify_names is not None:
        shap_df['modified_names'] = shap_df[name_key].map(modify_names)
    
    ism_df = cleaning(sample.lower(), apply_filter=False, outfile_pattern=ismoutfile_pattern)
    if modify_names is not None:
        ism_df['modified_names'] = ism_df[name_key].map(modify_names)
    
    if modify_names is not None:
        name_key = 'modified_names'
    ism_coi = [name_key, 'seqnames', 'start', 'end', 'es', 'enrichment', ]
    shap_coi = [name_key, 'seqnames', 'start', 'end', 'contribution', 'enrichment_shap', ]

    dt = ism_df[ism_coi].merge(shap_df[shap_coi], on=[name_key, 'seqnames', 'start', 'end'], how='outer')
    
    # add count before
    dt = dt.merge(dt.groupby(name_key).count()['end'], how='left', left_on=name_key, right_index=True, suffixes=['', '_cnt_before'])

    # add count after
    dt = dt.merge(dt.loc[(dt['enrichment_shap']) | (dt['enrichment']=='cleaned')].groupby(name_key).count()['end'], how='left', left_on=name_key, right_index=True, suffixes=['', '_cnt_after'])
    
    if verbose:
        print('  ... done after {:.1f}-s'.format(time.time() - tic))
        
    # return only cleaned?
    if return_only_cleaned:
        dt = dt.loc[(dt['enrichment_shap']) | (dt['enrichment']=='cleaned')]
        
    return dt
    
    
def merge_shap_ism_samples(
    samples: list = ['brca', 'blca', 'luad', 'kirp', 'kirc', 'skcm', 'gbm', 'coad', ],
    modify_names: dict or None = None,
    file_pattern: str = './tmp/TCGA/peak_shap/{}',
    ismoutfile_pattern: str ='./tmp/results/{}_v122.pkl',
    verbose: bool = True,
    outfile: str or None = None,
    modify_sample_name: bool = True,
):
    """Identify significant motifs in a sample
    
    Arguments: 
      modif_names: dict or None (optional, default: None)
        Modify the motif name with broader name sets by providing a dict
        key that will alter tha name_key to a new column `modified_names`, 
        then use that column to count things
    """
    if outfile is not None:
        if not os.path.exists(os.path.split(outfile)[0]):
            os.makedirs(os.path.split(outfile)[0])
    if verbose:
        tic = time.time()
        print('Merging samples:', samples)
    out = {}
    for s in samples:
        out[s] = merge_shap_ism(sample=s, file_pattern=file_pattern, 
                                modify_names=modify_names, 
                                ismoutfile_pattern=ismoutfile_pattern, 
                                modify_sample_name=modify_sample_name)
        if verbose:
            print('  through {}\telapsed: {:.1f}-s'.format(s, time.time() - tic))
    if outfile is not None:
        with open(outfile, 'wb') as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
    return out

def viz_compare_enrich(
    df: pd.DataFrame,
    sample_key: str = 'sample',
    name_key: str = 'group_name',
    drop_duplicates_key: str or None = 'group_name',
    topn: int = 10,
    fillna: bool = False,
    names2display: dict or None = None,
    viz_OR: bool = True,
    drop: str = None,
    zscore: str or None = None,
    save_plot: str or None = None,
    save_mat: str or None = None,
):
    """Visualize the top motifs in each sample in a heatmap
    
    Arguments:
      name_key
        will be used to index the visuzliation
      names2display: dict or None (optional, Default: None)
        send a dict to map the sample key to a new naming scheme for display
      drop: str or None (optional, default: None)
        ignore anything with this string pattern in the names
      zscore: str (optional, default: None)
        option to zscore 'row', 'col', or 'row' first, then 'col', i.e., 'rowcol'
    """
    
    agg = pd.DataFrame()
    valid_names = []
    for s in np.sort(df[sample_key].unique()):
        dt = df.loc[df[sample_key]==s, :]
        if drop is not None:
            dt = dt.loc[[False if drop in n else True for n in dt[name_key]], :]
        dt = dt.sort_values(by=['p_bonferonni', 'OR'], ascending=[True, False])
        if drop_duplicates_key is not None:
            dt = dt.drop_duplicates(subset=drop_duplicates_key)
        dt = dt.iloc[:topn, :] 
        if not fillna:
            valid_names += dt[name_key].unique().tolist()
        else:
            agg = pd.concat([agg, dt], ignore_index=True)
    if not fillna:
        valid_names = list(np.unique(valid_names))
        for s in np.sort(df[sample_key].unique()):
            dt = df.loc[df[sample_key]==s, :]
            dt = dt.sort_values(by=['p_bonferonni', 'OR'], ascending=[True, False])
            if drop_duplicates_key is not None:
                dt = dt.drop_duplicates(subset=drop_duplicates_key)
            dt = dt.loc[[True if i in valid_names else False for i in dt[name_key]], :]
            agg = pd.concat([agg, dt], ignore_index=True)
    
    
    if names2display is not None:
        agg['sample2'] = agg[sample_key].map(names2display)
        sample_key = 'sample2'   
    if False:
        dtt = agg.loc[[False if i.split('|')[1].startswith('Z') else True for i in agg[name_key]], :] # modify group_name
    else:
        dtt = agg
    if viz_OR:
        if False:
            # log2( OR )
            dt = (np.log2(pd.pivot(dtt, index=name_key, columns=sample_key)['OR']))
        else:
            # not transformed OR
            dt = (pd.pivot(dtt, index=name_key, columns=sample_key)['OR'])
    else:
        dt = (-1*np.log10(pd.pivot(dtt, index=name_key, columns=sample_key)['p_bonferonni']))
    if topn <= 10:
        figsize = (10, 10)
    if topn > 10:
        figsize = (10, 15)
    if topn > 20: 
        figsize = (10, 20)
    if fillna:
        dt.fillna(0, inplace=True)
    dt.replace([np.inf], 300, inplace=True)
    if zscore is None:
        p = sns.clustermap(dt, xticklabels=True, yticklabels=True, cmap='hot_r', row_cluster=True, col_cluster=True, figsize=figsize)
    elif zscore == 'col':
        p = sns.clustermap(dt, xticklabels=True, yticklabels=True, cmap='hot_r', row_cluster=True, col_cluster=True, z_score=1, figsize=figsize)
    elif zscore == 'row':
        p = sns.clustermap(dt, xticklabels=True, yticklabels=True, cmap='hot_r', row_cluster=True, col_cluster=True, z_score=0, figsize=figsize)
    elif zscore == 'rowcol':
        from scipy.stats import zscore
        dt = zscore(dt, axis=1)
        p = sns.clustermap(dt, xticklabels=True, yticklabels=True, cmap='hot_r', row_cluster=True, z_score=1, figsize=figsize)
    if save_plot is not None:
        p.savefig(save_plot)
    if save_mat is not None:
        p.data2d.to_csv(save_mat)
    return agg


# by arvind kumar for his analysis
def write_motifs_individually(cleaned_dict_pkl,motif_outdir):
    with open(cleaned_dict_pkl, 'rb') as handle:
        cleaned_motifs_dict = pickle.load(handle)
    if not os.path.exists(motif_outdir):
        os.makedirs(motif_outdir)
    for key,df in cleaned_motifs_dict.items():
        print("Processing: ",key)
        outfile = f"{motif_outdir}/{key.upper()}_motif_ism_shap_cleaned.csv"        
        df.to_csv(outfile,index=None)
        print("Saved to: ", outfile)
        write_motifs_individually(
            "./tmp/results/reg_diffs/ismshap_cancers_vierstrav1.pkl",
            "./tmp/TCGA/motifs/")
        
        
def chk_all_chunks(output_file_dir: str, motif_file: str or None = None, n_lines: int or None = None, chunksize=1e5):
    """
    Arguments:
      n_lines: int or None (optional, Default: None)
        if an int is fed, then skip loading and counting lines. If this is not 

    Examples:
      import sys
      sys.path.append('./reg_diffs/')
      from reg_diffs.scripts import evalism_v22 as evalism
      motifs = './tmp/vierstra_Archetype_BRCA_cancer_control_consensusagg.csv.bed'
      output_filepath = './results/brca10_indbrcavctrl_v122'
      chk = chk_all_chunks(output_file_dir=output_filepath, motif_file=motifs,)
      chk = chk_all_chunks(output_file_dir=output_filepath, n_lines=89456402, chunksize=1e5, )
    """
    if n_lines is None:
        # open and read file
        def blocks(file, size=65536):
            while True:
                b = file.read(size)
                if not b: break
                yield b

        with open(motif_file, "r",encoding="utf-8",errors='ignore') as f:
            n_lines = sum(bl.count("\n") for bl in blocks(f))
        print('n_lines:', n_lines)

    # get file list to comapre against 
    assert os.path.exists(output_file_dir), 'file dir does not exist!'   
    res_files = glob.glob(os.path.join(output_file_dir, '*chunk*.pkl'))
    assert len(res_files) > 0, 'output_file_dir exists but does not have files of *chunk*.pkl format'
    missing_chunks = []
    for i, ii in enumerate(range(0, n_lines, int(chunksize))): # don't need n_lines +1 because of header
        if not any([True if 'chunk{}.'.format(i) in fname else False for fname in res_files]):
            missing_chunks.append('chunk{}.pkl'.format(i))

    if len(missing_chunks) != 0:
        print('Missing motifs by lack of chunks:')
        # chk range of files 2 before and 2 after one in question to see if there is truly missing one
        for k, chunk in enumerate(missing_chunks):
            chunk_count = re.search('chunk([0-9]*).', chunk)[1]
            chunk_count = int(chunk_count)
            range_files = ['chunk{}.'.format(n) for n in range(max(0, chunk_count - 2), min(chunk_count + 3, n_lines))]
            surrounding_files = []
            for f in range_files:
                tmp = []
                for s in res_files:
                    if f in s:
                        tmp.append(s)
                surrounding_files.append(tmp)
            # remove nesting
            surrounding_files = [s for l in surrounding_files for s in l]
            print('  missing {}: {}'.format(k+1, chunk))
            print('    surrounding files:', surrounding_files)

            
    else:
        print('Not missing any motifs. Proceed.')


    return {'missing_chunks': missing_chunks, 'res_files': res_files}

# chk many
def chk_motifism(output_files: list, motif_file: str or None = None, n_lines: int or None = None, chunksize=1e5):
    """
    Arguments:
      n_lines: int or None (optional, Default: None)
        if an int is fed, then skip loading and counting lines. If this is not 

    Examples:
      import sys
      sys.path.append('./reg_diffs')
      from reg_diffs.scripts import evalism_v22 as evalism
      # motifs = './tmp/vierstra_Archetype_BRCA_cancer_control_consensusagg.csv.bed'
      
      # 1/ set up files, chk others' run 
      pfp = './results'
      suffix = '_indbrcavctrl_v122'
      samples = ['brca10', 'brca11', 'brca12', 'brca16','brca17', 'brca18', 'brca19', 'ctrl']
      output_files = [os.path.join(pfp, s+suffix) for s in samples]
      chk = evalism.chk_motifism(output_files=output_files, n_lines=89456402, chunksize=1e5, )
    """
    assert not (n_lines is None and motif_file is None), 'need to specify how to set up reader iterator'
    res = {}
    for filegrp in output_files:
        if n_lines is None:
            chk = chk_all_chunks(output_file_dir=filegrp, motif_file=motif_file, chunksize=chunksize, )
        else:
            chk = chk_all_chunks(output_file_dir=filegrp, n_lines=n_lines, chunksize=chunksize, )
        
        res[os.path.split(filegrp)[1]] = chk['missing_chunks']

    print('\nSummary:')
    for k, v in res.items():
        if len(v) > 0:
            print('\n  {} missing n_chnk={}:'.format(k, len(v)), v)
        else:
            print('\n  CLEAN result for {}'.format(k))
    return res
        

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--filepath', type=str,
                        help="filepath to chunked .pkl result files to process. Can specify sample shorthand too, e.g., brca10")
    args = parser.parse_args()

    print('\nArguments passed:')
    print(args)

    valid_file_specs = [
        'blca', 'skcm', 'gbm', 
        'kirp', 'kirc', 'luad', 'coad',

        'GBM39_cloneA', 'GBM39_cloneB', 
        'GBM45_cloneA', 'GBM45_cloneB',
        ]
    
    valid_file_specs = [s.lower() for s in valid_file_specs]
    
    if args.filepath.lower() in valid_file_specs or 'brca' in args.filepath.lower():
        g = args.filepath.lower() # note: motifism converts all to lowercase
        args.filepath = './tmp/reg_diffs/results/tmp/{}_v122/'.format(g)
        assert os.path.exists(args.filepath), 'point to valid filepath'
        outfile = './tmp/results/{}_v12.pkl'.format(g)

        df, null = main(
            filepath=args.filepath,
            outfile=outfile,
            )

        print('Done with {}. Outfile:'.format(g), outfile)
    
    elif args.filepath.lower() == 'merge_cancers':
        samples = ['coad', 'skcm', 'brca', 'luad', 'gbm', 'blca', 'kirc', 'kirp'] 

        enrich_df = merge_shap_ism_samples(
            samples=samples,
            modify_names=None,
            file_pattern='./tmp/TCGA/peak_shap/{}',
            ismoutfile_pattern='./tmp/results/{}_v12.pkl',
            outfile='./tmp/results/reg_diffs/ismshap_cancers_vierstrav1.pkl')

    elif args.filepath.lower() == 'merge_indbrca':

        # specify groups

        groups = {'luminal': ['brca{}'.format(n) for n in ['10', '12', '15', '20', '22', ]],
                'basal': ['brca{}'.format(n) for n in ['14', '16', '23', '24', '25', ]],
                'her2': ['brca{}'.format(n) for n in ['11', '13', '17', '18', '19', '21', ]]}

        sample2grp = {}
        for k, v in groups.items():
            for s in v:
                sample2grp[s] = '{}.{}'.format(k, s.split('brca')[-1])

        samples = [f"brca{x}" for x in range(10,26)]

        enrich_df = merge_shap_ism_samples(
            samples=samples,
            modify_names=None,
            file_pattern='./tmp/TCGA/peak_shap/brca_samples/{}',
            ismoutfile_pattern='./tmp/results/{}_v12.pkl',
            outfile='./tmp/results/reg_diffs/ismshap_indbrca_vierstrav1.pkl')


    elif args.filepath.lower() == 'merge_gbm':
        samples = ['GBM39_cloneA', 'GBM39_cloneB', 'GBM45_cloneA', 'GBM45_cloneB']

        enrich_df = merge_shap_ism_samples(
            samples=samples,
            modify_names=None,
            file_pattern='./tmp/TCGA/peak_shap/gbm_subclone/{}',
            ismoutfile_pattern='./tmp/results/{}_v12.pkl',
            outfile='./tmp/results/reg_diffs/ismshap_gbm_subclone_vierstrav1.pkl',
            modify_sample_name=False,)


