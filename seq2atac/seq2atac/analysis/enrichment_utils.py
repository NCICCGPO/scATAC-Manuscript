import numpy as np
import pandas as pd
from copy import deepcopy
from functools import reduce

import matplotlib.pyplot as plt
import scipy.stats

from seq2atac.stable import read_pickle
import seaborn as sns
def get_refalt_sequence(df,input_width,fasta_seq):
    ref_seqs = []
    alt_seqs = []
    
    for idx,row in df.iterrows():

        chrom,pos,ref,alt = row["Chromosome"], int(row["hg38_start"]), row["Reference_Allele"], row["Tumor_Seq_Allele2"]
        
        ### given: chrom, position, ref, alt, 
        ## get (2,input_width,4) => one with reference, one with alt

        half = input_width//2
        left = int(pos - half)
        right = int(pos + half)

        seq1 = str(fasta_seq[chrom][left:right])
        
        assert len(seq1) == input_width
        assert seq1[half] == ref

        seq2 = list(deepcopy(seq1))
        seq2[half]= alt
        seq2 = "".join(seq2)

        ## make sure the regions other than half-point is same
        assert seq1[0:half] == seq2[0:half]
        assert seq1[half+1:] == seq2[half+1:]

        ref_seqs.append(seq1)
        alt_seqs.append(seq2)
    
    return ref_seqs,alt_seqs


def get_alt_sequence(df,input_width,fasta_seq):
    
    alt_seqs = []
    peak_seqs = []
    
    for idx,row in df.iterrows():
        
        chm,peakstart,peakend=row["peak_chr"],row["peak_start"],row["peak_end"]
        ref,alt = row["Reference_Allele"], row["Tumor_Seq_Allele2"]
        
        peaksize = peakend - peakstart
        flank = (input_width - peaksize)//2
        peakstart = peakstart - flank
        peakend = peakstart + input_width
    
        peak_seq = str(fasta_seq[chm][peakstart:peakend])
        
        point = row["hg38_start"]
        dist_from_start = point - peakstart
        
        assert peak_seq[dist_from_start] == ref
        
        alt_seq = list(peak_seq)
        alt_seq[dist_from_start] = alt
        alt_seq = "".join(alt_seq)
        
        alt_seqs.append(alt_seq)
        peak_seqs.append(peak_seq)
        
    return peak_seqs, alt_seqs


def compute_ref_alt_scores(df,fold_score_files):
    
    assert len(fold_score_files) == 5
    proba_df = [read_pickle(f) for f in fold_score_files]
    proba_df = reduce(lambda x,y: x+y, proba_df)
    proba_df /= len(fold_score_files)
    assert not proba_df.isnull().values.any()
    
    return pd.concat([df,proba_df], axis=1)

def load_tracks(df,track_files):
    
    assert len(track_files) == 5
    tracks = [np.load(f) for f in track_files]
    tracks = reduce(lambda x,y: x+y, tracks)
    tracks /= len(track_files)
    assert len(tracks) == len(df)
    
    return tracks


def create_pancancer_distribution_plots(somatic_dfs,control_dfs,colname,cumulative,test="ranksums",transform=False,matplotlib=False):
    ### distributions
    fig,axes = plt.subplots(4,2,figsize=(10,10))
    axes = axes.ravel()
    df_stats = pd.DataFrame()
    for cidx,cancer_name in enumerate(somatic_dfs.keys()):
        ax = axes[cidx]
        
        c1 = somatic_dfs[cancer_name][colname]
        c2 = control_dfs[cancer_name][colname]

        if transform:
            c1=transform(c1)
            c2=transform(c2)

        
        numbins = 30
        if colname == "diff_summit_centered" or colname == "diff_mutation_centered":
            numbins = 100

        if matplotlib:
            ax.hist(c1,alpha=0.5,density=True,cumulative=cumulative,label="case",color="darkblue")
            ax.hist(c2,alpha=0.5,density=True,cumulative=cumulative,label="control",color="darkorange")

        # ax=sns.distplot(c1, hist=False, kde=True, bins=numbins, color = 'green',ax=ax,label="case",hist_kws={"alpha":0.5})
        # ax=sns.distplot(c2, hist=False, kde=True, bins=numbins, color = 'red',ax=ax,label="control",hist_kws={"alpha":0.5})
        else:
            ax=sns.kdeplot(c1, color='darkblue',ax=ax,label="case", alpha=0.5,cumulative=cumulative,fill=cumulative)
            ax=sns.kdeplot(c2, color='darkorange',ax=ax,label="control",alpha=0.5,cumulative=cumulative,fill=cumulative)


        if test == "ranksums":
            r,pval = scipy.stats.ranksums(c1,c2,"two-sided")
        elif test == "ttest":
            r,pval = scipy.stats.ttest_ind(c1,c2)
        elif test == "ks":
            r,pval = scipy.stats.ks_2samp(c1,c2,"two-sided")
        elif test == "wilcoxon":
            r,pval = scipy.stats.wilcoxon(c1,c2)     
        else:
            r,pval = scipy.stats.ranksums(c1,c2,"two-sided")
        ax.set_title(cancer_name + f" Ranksum: {round(r,2)} p-value: {round(pval,2)}")
        df_stats.loc[cancer_name,"case"] = len(c1)
        df_stats.loc[cancer_name,"control"] = len(c2)
        df_stats.loc[cancer_name,"case mean"] = c1.mean()
        df_stats.loc[cancer_name,"control mean"] = c2.mean()
        df_stats.loc[cancer_name,test] = r
        df_stats.loc[cancer_name,"pvalue"] = pval

    axes[7].set_visible(False)
    axes[6].legend(loc="upper right")
    fig.tight_layout()
    plt.legend()
    plt.show()

    return df_stats


def create_pancancer_distribution_plots_discrete(somatic_dfs,control_dfs,colname):
    ### distributions
    df_stats = pd.DataFrame()
    for cidx,cancer_name in enumerate(somatic_dfs.keys()):

        sdf = somatic_dfs[cancer_name].copy()
        cdf = control_dfs[cancer_name].copy()
        
        l1=sdf[colname].value_counts().reset_index().sort_values("index")[colname].tolist()
        l2=cdf[colname].value_counts().reset_index().sort_values("index")[colname].tolist()
        scipy.stats.kruskal(*l1,*l2)

        r,pval = scipy.stats.kruskal(*l1,*l2)
        df_stats.loc[cancer_name,"Somatic"] = len(sdf)
        df_stats.loc[cancer_name,"Matched gnomAD"] = len(cdf)
        df_stats.loc[cancer_name,"kruskal"] = r
        df_stats.loc[cancer_name,"pvalue"] = pval

    print(colname)
    return df_stats


def create_pancancer_correlations(dfs,col1,col2):
    ### distributions
    fig,axes = plt.subplots(4,2,figsize=(10,10))
    axes = axes.ravel()
    df_stats = pd.DataFrame()
    for cidx,cancer_name in enumerate(dfs.keys()):
        ax = axes[cidx]
        
        c1 = dfs[cancer_name][col1]
        c2 = dfs[cancer_name][col2]
        
        ax.scatter(c1,c2, s=1, color="Green",alpha=0.5)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        r,pval = scipy.stats.spearmanr(c1,c2)
        ax.set_title(cancer_name + f" spearman: {round(r,2)}")
        df_stats.loc[cancer_name,"counts"] = len(c1)
        df_stats.loc[cancer_name,"spearman"] = r
        df_stats.loc[cancer_name,"pvalue"] = pval
    fig.tight_layout()
    plt.legend()
    plt.show()
    return df_stats

def create_pancancer_valuecounts(dfs,colname):
    fig,axes = plt.subplots(4,2,figsize=(10,10))
    axes = axes.ravel()

    for cidx,cancer_name in enumerate(dfs.keys()):
        ax = axes[cidx]
        pd.DataFrame(dfs[cancer_name][colname].value_counts()).plot.pie(y=0,ax=ax)
        ax.set_title(cancer_name)
    plt.legend()
    plt.show()


def get_thresh(df,colname,percentile):
    return df[colname].quantile(percentile)

