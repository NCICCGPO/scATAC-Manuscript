import pandas as pd
import time
import pickle

import numpy as np
import pandas as pd
import time

import pickle

from scipy.stats import chi2_contingency, spearmanr

import re
import pybedtools
import scipy



def load_from_pickle(file: str or None, verbose:bool = True,):
    if verbose:
        print('Loading:', file)
        tic = time.time()
    with open(file, 'rb') as f:
        data =  pickle.load(f)
        f.close()
    if verbose:
        print('... done in {:.1f}-s'.format(time.time() - tic))
        if isinstance(data, dict):
            print('  keys:', data.keys())
        elif isinstance(data, pd.DataFrame):
            print('  df:', data.shape)
        else:
            print('  data type:', type(data))
    return data

def quick_enrichment_tst(a, b, c, d, verbose: bool = True,):
    """
    [a b]
    [c d]
    """
    table = np.array(
        [
         [a, b],
         [c, d],
        ])
    if verbose:
        print('\nContingency table:\n', ' ', table)
    OddsRatio = (table[0, 0] / table[0, 1]) / (table[1, 0] / table[1, 1])
    if verbose:
        print(f"OR: {OddsRatio:.4f}")
    chi2, p, dof, expected = chi2_contingency(table)
    if verbose:
        print(f"P-val: {p:.4e} ({p_val_enc(p)})")
    return (OddsRatio, p), table

def chi2ovr_cnt(
    data: dict,
    groups: dict or None = None,
    cnt_colkey: str = 'modified_names',
    verbose: bool = False,
    dev_mode: bool = False,
    include_tab: bool = False,
): 
    """
    Arguments:
      data: dict 
        dictionary of pd.DataFrames with sample_names as keys
    """
    
    if verbose: 
        tic = time.time()

    if include_tab:
        out = {cnt_colkey: [], 'p_chi2': [], 'OR': [], 'sample': [], 'p_bonferonni': [],
               'a': [], 'b': [], 'c': [], 'd': [], }
    else:    
        out = {cnt_colkey: [], 'p_chi2': [], 'OR': [], 'sample': [], 'p_bonferonni': []}
    count = 0
    for k in data.keys():
        if groups is not None:
            group = [g for g, v in groups.items() if k in v][0]
            background_keys = [kk for k, v in groups.items() for kk in v if kk not in groups[group]]
        else:
            background_keys = set(data.keys()) - set([k])
        
        # merge dataframes
        df = data[k]
        df['group_key'] = 'target'
        for kk in background_keys:
            dt = data[kk]
            dt['group_key'] = 'background'
            df = pd.concat([df, dt])
        
        # count
        target = df.loc[(df['group_key']=='target')].groupby(cnt_colkey).count().iloc[:, 0]
        bg = df.loc[(df['group_key']=='background')].groupby(cnt_colkey).count().iloc[:, 0]

        n_tsts = len(df[cnt_colkey].unique()) # for bonfernni
        for t in np.sort(df[cnt_colkey].unique()):
            if verbose:
                print('eval:', t)
            try: 
                a = target.loc[t].item()
            except KeyError:
                n_tsts = n_tsts - 1
                print('omitting {} from {}'.format(t, k))
                continue
            b = target.drop(t).sum()
            try:
                c = bg.loc[t].item()
            except KeyError:
                n_tsts = n_tsts - 1
                print('omitting {} from {}'.format(t, k))
                continue
            d = bg.drop(t).sum()
            if dev_mode:
                return a, b, c, d
            (OR, p), tab = quick_enrichment_tst(a, b, c, d, verbose=False)
            out[cnt_colkey].append(t)
            out['p_chi2'].append(p)
            out['OR'].append(OR)
            out['sample'].append(k)
            out['p_bonferonni'].append(min(1.0, p*n_tsts))
            if include_tab:
                out['a'].append(tab[0, 0])
                out['b'].append(tab[0, 1])
                out['c'].append(tab[1, 0])
                out['d'].append(tab[1, 1])
    
    return pd.DataFrame(out)

def left_intersect(dfA, dfB, 
                   A_coloi=['seqnames', 'start', 'end', ],
                   B_coloi=['seqnames', 'start', 'end'],):
    A_coloi += [c for c in dfA.columns if c not in A_coloi]
    A = pybedtools.BedTool.from_dataframe(dfA[A_coloi])
    
    # CNV
    B_coloi = B_coloi + [c for c in dfB.columns if c not in B_coloi]
    B = pybedtools.BedTool.from_dataframe(dfB[B_coloi])
    
    C = A.intersect(B, wa=True)
    # targets = [a + '_x' if a in B_coloi else a for a in A_coloi] + [b + '_y' if b in A_coloi else b for b in B_coloi]
    C = C.to_dataframe(names=A_coloi)#targets)
    C.index = C.index.astype(str)
    return C

def peakset2df(input_dict,):
    out = {}
    for k, v in input_dict.items():
        chrom, start, end = [], [], []
        for s in v:
            s = s.split(',_,')
            chrom.append(s[0])
            start.append(s[1])
            end.append(s[2])
        out[k] = pd.DataFrame({'seqnames':chrom, 'start': start, 'end': end})
    return out
    
    
def mtfcmp_indbrcavctrl_v2(
    cleaned_motif_file: str,
    peakset_file: str, 
):

    name_key = 'group_name' 
    include_tab = True

    ## get active motifs
    tmpdata = load_from_pickle(cleaned_motif_file)

    ## rename keys
    newname = {
        'brca10_indbrcavctrl': 'LumA_1', 'brca11_indbrcavctrl': 'HER2_5', 'brca12_indbrcavctrl': 'LumB_4', 
        'brca13_indbrcavctrl': 'HER2_4', 'brca14_indbrcavctrl': 'BASAL1', 'brca15_indbrcavctrl': 'LumA_2', 
        'brca16_indbrcavctrl': 'BASAL2', 'brca17_indbrcavctrl': 'HER2_1', 'brca18_indbrcavctrl': 'HER2_1a', 
        'brca19_indbrcavctrl': 'HER2_3', 'brca20_indbrcavctrl': 'LumA_3', 'brca21_indbrcavctrl': 'HER2_2', 
        'brca22_indbrcavctrl': 'LumB_5', 'brca23_indbrcavctrl': 'BASAL3', 'brca24_indbrcavctrl': 'BASAL5', 
        'brca25_indbrcavctrl': 'BASAL4', 'ctrl_indbrcavctrl': 'Ctrl'}
    data = {}
    for k in tmpdata.keys():
        data[newname[k]] = tmpdata[k]
    del tmpdata

    ## get peaksets
    with open(peakset_file, 'rb') as f:
        peaksets = pickle.load(f)
        f.close()


    ## UP vs. DOWN

    ### intersect
    res = {}
    tic = time.time()
    samples2eval = list(data.keys())
    samples2eval.sort()
    for i, k in enumerate(samples2eval):
        # ignore ctrl
        if k != 'Ctrl':
            #### get list of unique differential pks
            print('\nstarting:', k, '...')
            try: 
                pks = peakset2df(peaksets[k])
            except KeyError :
                print('SKIPPING:', k, 'due to absence in diff pk set')
                continue
            
            for tst in ['tstup', 'tstdo']:
                #### intersect active motif file with pks file
                dt = {}
                
                if tst == 'tstup':
                    print('  pre-intersect tst up: {} motifs:'.format(k), data[k].shape)
                    print('  pre-intersect tst up: {} pks:'.format(k), pks['up'].shape)

                    dt['{}_canc_cancup'.format(k)] = left_intersect(data[k].copy(), pks['up'].copy())
                    dt['{}_ctrl_cancup'.format(k)] = left_intersect(data['Ctrl'].copy(), pks['up'].copy())

                if tst == 'tstdo':
                    print('  pre-intersect tst do: {} motifs:'.format(k), data[k].shape)
                    print('  pre-intersect tst do: {} pks:'.format(k), pks['do'].shape)

                    dt['{}_canc_cancdo'.format(k)] = left_intersect(data[k].copy(), pks['do'].copy())
                    dt['{}_ctrl_cancdo'.format(k)] = left_intersect(data['Ctrl'].copy(), pks['do'].copy())
                
                for kk,v in dt.items():
                    print('post-intersect:', kk, v.shape)
                
                enrich_out = chi2ovr_cnt(
                    data=dt,
                    cnt_colkey=name_key,
                    include_tab=include_tab,
                )

                # retain botu up and down
                res[k + '_' + tst] = enrich_out # .loc[enrich_out['sample']=='{}_up'.format(k)]
            print('... done:', k, 'elapsed: {:.1f}'.format(time.time() - tic))

    ### merge all into one dataframe
    df = pd.DataFrame()
    for k, v in res.items():
        df = pd.concat([df, v])    

    return df


def viz_mtfcmp(
    df: pd.DataFrame,
    sample_key: str = 'sample',
    name_key: str = 'group_name',
    drop_duplicates_key: str or None = None,
    topn: int or None or str = 10,
    fillna: bool = False,
    names2display: dict or None = None,
    viz_OR: bool = True,
    drop: str = None,
    cmap: str = 'hot_r',
    zscore: str or None = None,
    save_plot: str or None = None,
    save_mat: str or None = None,
    return_plot_obj: bool = False,
    col_cluster: bool = True,
    figsize_wh = (12, 20),
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
    valid_names = []
    df = df.sort_values(by=[sample_key, 'p_bonferonni', 'OR'], ascending=[True, True, False])
    if drop is not None:
        df = df.loc[[False if drop in n else True for n in df[name_key]], :]
    if drop_duplicates_key is not None:
        # this is pretty error prone
        df = df.drop_duplicates(subset=[sample_key, drop_duplicates_key])


    # keep only rows with significant hits
    dt = df.groupby([name_key])['p_bonferonni'].min().reset_index()
    for n in dt.loc[dt['p_bonferonni'] < 0.01, name_key].to_list():
        valid_names.append(n)
    df = df.loc[[True if n in valid_names else False for n in df[name_key]], :]


    if topn=='sig_only_top10OR':
        # top 10 lt1 and top 10 gt1 per sample
        # reset valid names
        valid_names = []
        for s in df[sample_key].unique():
            dt = df.loc[df[sample_key]==s, :].sort_values(by=['OR'], ascending=[True]).iloc[:10, :]
            for n in dt[name_key]:
                valid_names.append(n)
            dt = df.loc[df[sample_key]==s, :].sort_values(by=['OR'], ascending=[False]).iloc[:10, :]
            for n in dt[name_key]:
                valid_names.append(n)
        # filter 
        df = df.loc[[True if n in valid_names else False for n in df[name_key]], :]

    elif topn is not None and topn=='sig_only_top10ORgt1':
        # top 10 lt1 and top 10 gt1 per sample
        # reset valid names
        valid_names = []
        for s in df[sample_key].unique():
            dt = df.loc[df[sample_key]==s, :].sort_values(by=['OR'], ascending=[False]).iloc[:10, :]
            for n in dt[name_key]:
                valid_names.append(n)
        # filter 
        df = df.loc[[True if n in valid_names else False for n in df[name_key]], :]

    else:
        assert False, "Wrong n choice"

    if names2display is not None:
        df['sample2'] = df[sample_key].map(names2display)
        sample_key = 'sample2'

    if viz_OR:
        if False:
            # log2( OR )
            dt = (np.log2(pd.pivot(df, index=name_key, columns=sample_key)['OR']))
        else:
            # not transformed OR
            dt = (pd.pivot(df, index=name_key, columns=sample_key)['OR'])
    else:
        dt = (-1*np.log10(pd.pivot(df, index=name_key, columns=sample_key)['p_bonferonni']))
        dt.replace([np.inf], 300, inplace=True)
    if fillna:
        dt.fillna(0, inplace=True)

    if isinstance(topn, str) and 'clipOR' in topn:
        thresh = re.findall('clipOR(\d*)', topn)
        assert len(thresh) == 1, 'only one number for clip'
        thresh = float(thresh[0])
        print('thresh:', thresh)
        # dt.loc[dt>=thresh] = thresh
    else: 
        thresh = None
    

    from scipy.stats import zscore
    dt = zscore(dt, axis=1)
    return {'df_plot': dt, 'df_prez': df}, None

