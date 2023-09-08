import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

compute_pval_star = lambda pval: "*" if pval < 0.05 else ""

def get_lof_fold_change(df1,df2,colname,thresh):

    l1 = df1[df1[colname]>thresh]
    l2 = df1[df1[colname]<=thresh]
    l3 = df2[df2[colname]>thresh]
    l4 = df2[df2[colname]<=thresh]

    fisher_matrix = np.array([[len(l1),len(l2)],[len(l3),len(l4)]])
    fold_change,pval = scipy.stats.fisher_exact(fisher_matrix)
    return fisher_matrix,fold_change,pval

def get_gof_fold_change(df1,df2,colname,thresh):

    l1 = df1[df1[colname]<thresh]
    l2 = df1[df1[colname]>=thresh]
    l3 = df2[df2[colname]<thresh]
    l4 = df2[df2[colname]>=thresh]

    fisher_matrix = np.array([[len(l1),len(l2)],[len(l3),len(l4)]])
    fold_change,pval = scipy.stats.fisher_exact(fisher_matrix)
    return fisher_matrix,fold_change,pval

def single_cancer_motif_he_enrichment(somatic_df,control_df, colname, thresh):

    l1 = somatic_df[(somatic_df[colname]>thresh) & (somatic_df["motif_hit"]!=0)]
    l2 = somatic_df[~somatic_df.index.isin(l1.index)]
    l3 = control_df[(control_df[colname]>thresh) & (control_df["motif_hit"]!=0)]
    l4 = control_df[~control_df.index.isin(l3.index)]

    fisher_matrix = np.array([[len(l1),len(l2)],[len(l3),len(l4)]])
    # fisher_matrix[fisher_matrix==0] = 1
    fold_change,pval = scipy.stats.fisher_exact(fisher_matrix)   
    
    return fisher_matrix,fold_change,pval

def plot_bars_single(ax,values,texts,label,shift,width):
    assert len(values) == len(texts)
    rectangles = ax.bar(np.arange(len(values)) - shift,
                        values,
                        width,
                        label=label)
    for rect_idx,rect in enumerate(rectangles):
        ax.text(rect.get_x() + rect.get_width() / 2.0, 
                rect.get_height(), 
                texts[rect_idx], 
                ha='center', 
                va='bottom')
    return ax
        
def plot_bars(values_dict,texts_dict,X_labels,title,plot_hline=1):
    
    ### values_dict: {key:[]}
    ### assert: each value of the dict is a list. under each key, the list order is assumed to be the same
    ### ie: if each list records sth for each cancer, this must be consistent across the "keys"
    ### texts_dict: similar to values_dict
    
    
    fig,ax = plt.subplots()
    X_axis = np.arange(len(X_labels))
    
    width = 1.0 / (len(values_dict) + 2) ### split the 1 unit interval into number of colors + 2
    shift = 0
    
    for label,values in values_dict.items():
        texts = texts_dict[label]
        
        ax = plot_bars_single(ax,values,texts,label,shift,width)
        shift -= width
        
    ax.set_xticks(X_axis, X_labels)
    if plot_hline:
        ax.axhline(plot_hline)
    ax.set_xlabel("Cancer type")
    ax.set_ylabel("Fold change")
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.legend()
    plt.show()


def run_enrichment(somatic_dict, control_dict, expt_title, fisher_function, thresholds = None):

    ### enrichment tests - case vs control | high effect vs not
    ### pancan vs non-pancan enrichment tests
    ### high_effect: determined from case 95th percentile

    ### thresholds: dict of dict. outer indexed by quantile, inner indexed by keys of somatic_dict

    colname = "diff_summit_centered"

    quantiles = []
    if not thresholds:
        quantiles = [0.95,0.975,0.99]
    else:
        quantiles = sorted(list(thresholds.keys()))

    all_tcga_cancers = list(somatic_dict.keys())
    print(all_tcga_cancers)
    
    ### TODO: We want (quantile,cancer_name) as keys
    fold_changes = {k:[] for k in quantiles}
    pvals = {k:[] for k in quantiles}
    pvals_raw = {k:[] for k in quantiles}
    fisher_matrices = {k:[] for k in quantiles}

    if not thresholds:
        thresholds = {}
        for q in quantiles:
            thresh_quant_dict = {cn:somatic_dict[cn][colname].quantile(q) for cn in all_tcga_cancers}
            thresholds[q] = thresh_quant_dict
    print(thresholds)

    for quantile in quantiles:
        print(quantile)
        for cancer_name in all_tcga_cancers:
            print(cancer_name)
            thresh = thresholds[quantile][cancer_name]
            print(somatic_dict[cancer_name].shape)
            print("thresh: ",thresh)

            fisher_matrix,fold_change,pval = fisher_function(somatic_dict[cancer_name],
                                                             control_dict[cancer_name],
                                                             colname,
                                                             thresh)
            print(fisher_matrix, fold_change, pval)
            fold_changes[quantile].append(fold_change)
            fisher_matrices[quantile].append(fisher_matrix)
            pval_star = compute_pval_star(pval)
            pvals[quantile].append(compute_pval_star(pval))
            pvals_raw[quantile].append(compute_pval_star(pval))


    plot_bars(fold_changes,pvals,all_tcga_cancers,expt_title)
    return fisher_matrices,fold_changes, pvals_raw


def run_enrichment_gof(somatic_dict, control_dict, expt_title, fisher_function, thresholds = None):

    ### enrichment tests - case vs control | high effect vs not
    ### pancan vs non-pancan enrichment tests
    ### high_effect: determined from case 95th percentile

    ### thresholds: dict of dict. outer indexed by quantile, inner indexed by keys of somatic_dict

    colname = "diff_summit_centered"

    quantiles = []
    if not thresholds:
        quantiles = [0.05,0.025,0.01]
    else:
        quantiles = sorted(list(thresholds.keys()))

    all_tcga_cancers = list(somatic_dict.keys())
    print(all_tcga_cancers)
    
    ### TODO: We want (quantile,cancer_name) as keys
    fold_changes = {k:[] for k in quantiles}
    pvals = {k:[] for k in quantiles}
    fisher_matrices = {k:[] for k in quantiles}

    if not thresholds:
        thresholds = {}
        for q in quantiles:
            thresh_quant_dict = {cn:somatic_dict[cn][colname].quantile(q) for cn in all_tcga_cancers}
            thresholds[q] = thresh_quant_dict
    print(thresholds)

    for quantile in quantiles:
        print(quantile)
        for cancer_name in all_tcga_cancers:
            print(cancer_name)
            thresh = thresholds[quantile][cancer_name]
            print(somatic_dict[cancer_name].shape)
            print("thresh: ",thresh)

            fisher_matrix,fold_change,pval = fisher_function(somatic_dict[cancer_name],
                                                             control_dict[cancer_name],
                                                             colname,
                                                             thresh)
            print(fisher_matrix, fold_change, pval)
            fold_changes[quantile].append(fold_change)
            fisher_matrices[quantile].append(fisher_matrix)
            pval_star = compute_pval_star(pval)
            pvals[quantile].append(compute_pval_star(pval))


    plot_bars(fold_changes,pvals,all_tcga_cancers,expt_title)
    return fisher_matrices,fold_changes, pvals

""""
### Deprecated code:

def run_enrichment_test(somatic_dict, control_dict, expt_title):

    ### enrichment tests - somatic vs gnomad | high effect vs not
    ### pancan vs non-pancan enrichment tests

    fig, axes = plt.subplots()
    colname = "diff_summit_centered"
    X_labels = list(somatic_dict.keys())
    X_axis = np.arange(len(X_labels))

    quantile_toshift_dict = [-0.2,0,0.2]
    for q_idx,quantile in enumerate([0.95,0.975,0.99]):
        print(quantile)
        quantile_scores = []
        pvalues = []
        for cidx,cancer_name in enumerate(X_labels):
            print(cancer_name)
            thresh = somatic_dict[cancer_name][colname].quantile(quantile)
            print(somatic_dict[cancer_name].shape)
            print("thresh: ",thresh)

            fisher_matrix,fold_change,pval = get_lof_fold_change(somatic_dict[cancer_name],
                                                                 control_dict[cancer_name],
                                                                 colname,
                                                                 thresh)
            print(fisher_matrix, fold_change, pval)
            quantile_scores.append(fold_change)
            pval_star = compute_pval_star(pval)
            pvalues.append(pval_star)

        quantile_toshift = quantile_toshift_dict[q_idx]
        rectangles = axes.bar(X_axis + quantile_toshift,quantile_scores,0.2,label=quantile)
        for rect_idx,rect in enumerate(rectangles):
            height = rect.get_height()
            axes.text(rect.get_x() + rect.get_width() / 2.0, height, pvalues[rect_idx], ha='center', va='bottom')
    axes.set_xticks(X_axis, X_labels)
    axes.axhline(1.0)
    fig.suptitle(expt_title)
    fig.tight_layout()
    plt.legend()
    plt.show()

def run_motif_enrichment_test(somatic_dict, control_dict, quantile, expt_title):

    ### enrichment tests - somatic vs gnomad | high effect vs not
    ### pancan vs non-pancan enrichment tests

    fig, axes = plt.subplots()
    colname = "diff_summit_centered"
    X_labels = list(somatic_dict.keys())
    X_axis = np.arange(len(X_labels))
    
    
    ### build thresholds
    thresholds = {cancer_name:somatic_dict[cancer_name]["diff_summit_centered"].quantile(quantile) for cancer_name in X_labels}
    print(thresholds)
    
    ### somatic bars
    quantile_toshift = 0
    quantile_scores = []
    pvalues = []
    for cidx,cancer_name in enumerate(X_labels):
        print(cancer_name)
        thresh = thresholds[cancer_name]
        print("thresh: ",thresh)
        
        somatic_df = somatic_dict[cancer_name].copy()

        fisher_matrix,fold_change,pval = get_lof_fold_change(somatic_df[somatic_df["motif_hit"]!=0],
                                                             somatic_df[somatic_df["motif_hit"]==0],
                                                             colname,
                                                             thresh)
        print(fisher_matrix, fold_change, pval)
        quantile_scores.append(fold_change)
        pval_star = compute_pval_star(pval)
        pvalues.append(pval_star)
        
    rectangles = axes.bar(X_axis + quantile_toshift,quantile_scores,0.2,label="somatic")
    for rect_idx,rect in enumerate(rectangles):
        height = rect.get_height()
        axes.text(rect.get_x() + rect.get_width() / 2.0, height, pvalues[rect_idx], ha='center', va='bottom')
        
#     ### build thresholds
#     thresholds = {cancer_name:control_dict[cancer_name]["diff_summit_centered"].quantile(quantile) for cancer_name in X_labels}
#     print(thresholds)
    
    ### control bars
    quantile_toshift = -0.2
    quantile_scores = []
    pvalues = []
    for cidx,cancer_name in enumerate(X_labels):
        print(cancer_name)
        thresh = thresholds[cancer_name]
        print("thresh: ",thresh)
        
        control_df = control_dict[cancer_name].copy()

        fisher_matrix,fold_change,pval = get_lof_fold_change(control_df[control_df["motif_hit"]!=0],
                                                             control_df[control_df["motif_hit"]==0],
                                                             colname,
                                                             thresh)
        print(fisher_matrix, fold_change, pval)
        quantile_scores.append(fold_change)
        pval_star = compute_pval_star(pval)
        pvalues.append(pval_star)
        
    rectangles = axes.bar(X_axis + quantile_toshift,quantile_scores,0.2,label="control")
    for rect_idx,rect in enumerate(rectangles):
        height = rect.get_height()
        axes.text(rect.get_x() + rect.get_width() / 2.0, height, pvalues[rect_idx], ha='center', va='bottom')
        

    axes.set_xticks(X_axis, X_labels)
    axes.axhline(1.0)
    fig.suptitle(expt_title)
    fig.tight_layout()
    plt.legend()
    plt.show()

def run_motif_enrichment_high_effect(somatic_dict, control_dict, quantile, expt_title):

    ### enrichment tests - somatic vs gnomad | high effect vs not
    ### pancan vs non-pancan enrichment tests

    fig, axes = plt.subplots()
    colname = "diff_summit_centered"
    X_labels = list(somatic_dict.keys())
    X_axis = np.arange(len(X_labels))

    thresholds = {cancer_name:somatic_dict[cancer_name]["diff_summit_centered"].quantile(quantile) for cancer_name in X_labels}
    print(thresholds)
    
    ### somatic bars
    quantile_toshift = 0
    quantile_scores = []
    pvalues = []
    for cidx,cancer_name in enumerate(X_labels):
        print(cancer_name)
        thresh = thresholds[cancer_name]
        print("thresh: ",thresh)
        
        somatic_df = somatic_dict[cancer_name].copy()
        control_df = control_dict[cancer_name].copy()
        
        
        l1 = somatic_df[(somatic_df[colname]>thresh) & (somatic_df["motif_hit"]!=0)]
        l2 = somatic_df[(somatic_df[colname]>thresh) & (somatic_df["motif_hit"]==0)]
        l3 = control_df[(control_df[colname]>thresh) & (control_df["motif_hit"]!=0)]
        l4 = control_df[(control_df[colname]>thresh) & (control_df["motif_hit"]==0)]
        
        fisher_matrix = np.array([[len(l1),len(l2)],[len(l3),len(l4)]])
        fisher_matrix[fisher_matrix==0] = 1
        fold_change,pval = scipy.stats.fisher_exact(fisher_matrix)
        
        print(fisher_matrix, fold_change, pval)
        quantile_scores.append(fold_change)
        pval_star = compute_pval_star(pval)
        pvalues.append(pval_star)
        
    rectangles = axes.bar(X_axis + quantile_toshift,quantile_scores,0.2,label="fold change")
    for rect_idx,rect in enumerate(rectangles):
        height = rect.get_height()
        axes.text(rect.get_x() + rect.get_width() / 2.0, height, pvalues[rect_idx], ha='center', va='bottom')


    axes.set_xticks(X_axis, X_labels)
    axes.axhline(1.0)
    fig.suptitle(expt_title)
    fig.tight_layout()
    plt.legend()
    plt.show()

"""