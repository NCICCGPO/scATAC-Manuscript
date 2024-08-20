import numpy as np
import pandas as pd
import time
import seaborn as sns
from scipy.stats import chi2_contingency


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

    
def viz_compare_enrich(
    df: pd.DataFrame,
    sample_key: str = 'sample',
    name_key: str = 'group_name',
    drop_duplicates_key: str or None = 'group_name',
    topn: int or None = 10,
    fillna: bool = False,
    names2display: dict or None = None,
    viz_OR: bool = True,
    drop: str = None,
    zscore: str or None = None,
    save_plot: str or None = None,
    save_mat: str or None = None,
    return_plot_obj: bool = False,
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
        if topn is not None:
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
    if fillna:
        dt.fillna(0, inplace=True)
    dt.replace([np.inf], 300, inplace=True)
    
    dtt = dt.copy()

    if zscore == 'rowcol':
        from scipy.stats import zscore
        dt = zscore(dt, axis=1)
        p = sns.clustermap(dt, xticklabels=True, yticklabels=True, cmap='hot_r', row_cluster=True, z_score=1)

    if return_plot_obj:
        return agg, p, {'df_plot': dt, 'df_prez': dtt}
    else:
        return agg