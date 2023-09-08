import pandas as pd
import numpy as np
from liftover import get_lifter
from pybedtools import BedTool
import scipy.stats

converter = get_lifter('hg19', 'hg38')

from seq2atac.analysis import gnomad_file, fasta_seq, refcomplement, sizesfile, get_promoterai_tss, get_cosmic_pancan_genes


def hg19_to_hg38(all_mutations,model_input_size = 1364):
    # all mutations is a file with a df with Chromosome, Start_position in hg19, Reference_Allele, Tumor_Seq_Allele2

    ### read the input file
    print("reading input file")
    df_mutations = pd.read_csv(all_mutations)

    ### 0 indexed position of mutation
    df_mutations['hg19_start']=(df_mutations['Start_position']-1).astype(int)

    ### liftover to hg38
    print("lifting over")
    new_ref_allele=[]
    new_strand_list=[]

    for idx,row in df_mutations.iterrows():
        if idx%10000 == 1:
            print(idx)
        chm = row["Chromosome"]
        hg19_start = row["hg19_start"]
        z=converter[chm][hg19_start]
        #print(z)
        try:
            new_Start=z[0][1]
            new_strand=z[0][2]
            new_ref_allele.append(new_Start)
            new_strand_list.append(new_strand)
        except:
            new_ref_allele.append('notfound')
            new_strand_list.append('notfound')

    df_mutations['hg38_start']=new_ref_allele
    df_mutations['hg38_strand']=new_strand_list


    #### Remove those that didnt get lifted over
    df_mutations=df_mutations[df_mutations['hg38_start']!='notfound'].reset_index(drop=True)
    print("Removed those that cannot be lifted over to hg38")
    print(df_mutations.shape)
    df_mutations['hg38_end']=(df_mutations['hg38_start']+1).astype(int)

    # #### Remove those that map to - strand in hg38
    df_mutations = df_mutations[df_mutations.hg38_strand=="+"].reset_index(drop=True)
    df_mutations = df_mutations.drop("hg38_strand",axis=1)
    print("Removed those mapping to - strand")
    print(df_mutations.shape)


    #### Remove those that dont have correct Ref
    correct_ref_indicator = []
    for idx,row in df_mutations.iterrows():
        if idx%10000 == 1:
            print(idx)
        chrom,pos,ref = row["Chromosome"], int(row["hg38_start"]), row["Reference_Allele"]

        if fasta_seq[chrom][pos] != ref:
            correct_ref_indicator.append(0)
        else:
            correct_ref_indicator.append(1)
    df_mutations["correct_ref_indicator"] = correct_ref_indicator

    df_mutations = df_mutations[df_mutations["correct_ref_indicator"] == 1].reset_index(drop=True)
    print("Removed those with incorrect ref in hg38")
    df_mutations = df_mutations.drop("correct_ref_indicator",axis=1)
    print(df_mutations.shape)

    #### Remove those that cant be flanked
    flankable_indicator = []
    for idx,row in df_mutations.iterrows():
        if idx%10000 == 1:
            print(idx)
        chrom,pos = row["Chromosome"], int(row["hg38_start"])

        half = model_input_size//2
        left = int(pos - half)
        right = int(pos + half)


        seq1 = str(fasta_seq[chrom][left:right])

        if len(seq1) != model_input_size:
            flankable_indicator.append(0)
        else:
            flankable_indicator.append(1)
    df_mutations["flankable_indicator"] = flankable_indicator

    df_mutations = df_mutations[df_mutations["flankable_indicator"] == 1].reset_index(drop=True)
    print(f"Removed those that cant be flanked to {model_input_size}")
    df_mutations = df_mutations.drop("flankable_indicator",axis=1)
    print(df_mutations.shape)

    return df_mutations

def compute_gnomad_membership(df):

    ### add 2 columns to df
    ## df: mutations format
    ## must have Chromosome, hg38_start, hg38_end, Reference_Allele, Tumor_Seq_Allele2

    gn = pd.read_csv(gnomad_file,sep="\t")
    gn["start"] = gn["pos"] - 1
    gn.drop("pos",axis=1,inplace=True)
    print(gn.shape)

    df_mutations = pd.merge(df, gn, \
                    how="left", \
                    left_on=["Chromosome","hg38_start","Tumor_Seq_Allele2"], \
                    right_on=["chrom","start","alt"])
    df_mutations.drop(["chrom","start","alt"],axis=1,inplace=True)
    print(df_mutations.shape)

    df_mutations["status"] = df_mutations["status"].fillna("NA") ## TODO: Change :NA: to :ABSENT:
    df_mutations["gnomad_af"] = df_mutations["gnomad_af"].fillna(0.0)
    print(df_mutations.shape)

    # df_mutations.to_csv(f"{outdir}/{cancer_name}_mutation_scores.tsv",sep="\t",index=None)
    return df_mutations

def get_mutation_type(ref,alt):
    if ref in ["C","T"]: ## CA, CG, CT, TA, TC, TG
        return ref+alt
    else: #AC,AG,AT,GA,GC,GT
        ref_rev = refcomplement[ref]
        alt_rev = refcomplement[alt]
        return ref_rev+alt_rev

def compute_trinuc(chr,start,end,ref,alt):
    predicted_ref = str(fasta_seq[chr][start])
    assert predicted_ref.upper() == ref.upper()
    left = str(fasta_seq[chr][start-1])
    right = str(fasta_seq[chr][start+1])
    mid = get_mutation_type(ref,alt)
    trinuc = left + mid + right
    trinuc = trinuc.upper()
    return trinuc

def compute_motif_hits(df,cleaned_motifs_file,slop=0,colname="motif_hit"):

    ## df: a bed format dataframe: first 3 columns must be chr, start, end
    ## cleaned_motifs file: a file with seqnames, start, end columns which correspond to cleaned motif instances
    ## colname: what is the name of the column added to the dataframe

    cleaned_motifs = pd.read_csv(cleaned_motifs_file)
    
    cleaned_bed= BedTool.from_dataframe(cleaned_motifs[["seqnames","start","end","group_name"]]).slop(b=slop,g=sizesfile)
    mutations_bed = BedTool.from_dataframe(df)
    
    target_columns = list(df.columns) + [colname]
    mutations_merged = mutations_bed.intersect(cleaned_bed,c=True).to_dataframe(names=target_columns)
    #mutations_merged = mutations_merged.rename(columns=dict(zip(mutations_merged.columns,target_columns)))
    return mutations_merged

def intersect_motif_file(df,cleaned_motifs_file,colname_begin):

    ## df: a bed format dataframe: first 3 columns must be chr, start, end
    ## cleaned_motifs file: a file with seqnames, start, end columns which correspond to cleaned motif instances
    ## colname: what is the name of the column added to the dataframe

    cleaned_motifs = pd.read_csv(cleaned_motifs_file)
    
    motif_cols = ["seqnames","start","end","group_name"]
    cleaned_bed= BedTool.from_dataframe(cleaned_motifs[motif_cols])
    mutations_cols = ["Chromosome","hg38_start","hg38_end","mutation_id"]
    mutations_bed = BedTool.from_dataframe(df[mutations_cols])
    
    target_columns = mutations_cols + [f"{colname_begin}_{x}" for x in motif_cols] + [f"{colname_begin}_motif_hits"]
    mutations_merged = mutations_bed.intersect(cleaned_bed,wao=True).to_dataframe(names=target_columns)
    #mutations_merged = mutations_merged.rename(columns=dict(zip(mutations_merged.columns,target_columns)))
    shape_before = mutations_merged.shape
    mutations_merged = pd.merge(mutations_merged,df,how="left",on=mutations_cols)
    shape_after = mutations_merged.shape
    assert shape_before[0] == shape_after[0]
    return mutations_merged


def compute_gene_annotation(df,genelist,kbp,column_name):

    ## df: a bed format dataframe: first 3 columns must be chr, start, end
    ## genelist file: list of genes of interest
    ## kbp: distance from tss's of each gene in the gene list to create the intervals
    ## column_name: what is the name of the column added to the dataframe

    tss_df = get_promoterai_tss()
    tss_df_genelist = tss_df[tss_df["gene"].isin(genelist)]
    tss_bed = BedTool.from_dataframe(tss_df_genelist).slop(b=int(kbp*1000),g=sizesfile)
    
    mutations_bed = BedTool.from_dataframe(df)
    target_columns = list(df.columns) + [column_name]
    mutations_merged = mutations_bed.intersect(tss_bed, c=True).to_dataframe(names=target_columns)
    #mutations_merged = mutations_merged.rename(columns=dict(zip(mutations_merged.columns,target_columns)))
    mutations_merged[column_name] = mutations_merged[column_name].apply(lambda x: int(x>=1))
    return mutations_merged

def compute_closest_gene_annotation(df,genelist):

    ## df: a bed format dataframe:  must have "Chromosome","hg38_start","hg38_end","mutation_id"
    ## genelist: list of genes of interest
    ### for each mutation, compute closest gene from genelist

    tss_df = get_promoterai_tss()
    tss_df_genelist = tss_df[tss_df["gene"].isin(genelist)]
    tss_bed = BedTool.from_dataframe(tss_df_genelist)
    
    selected_cols = ["Chromosome","hg38_start","hg38_end","mutation_id"]

    mutations_bed = BedTool.from_dataframe(df[selected_cols]) ## has 4 columns
    target_columns = selected_cols + ["gene_chr","gene_start","gene_end","gene"] + ["distance_to_tss"] 
    mutations_merged = mutations_bed.sort().closest(tss_bed.sort(), d=True).to_dataframe(names=target_columns) ### has 9 columns
    mutations_merged = mutations_merged.drop(["gene_chr","gene_start","gene_end"],axis=1).sort_values(selected_cols) ## has 6 columns
    mutations_merged = BedTool.from_dataframe(mutations_merged).groupby(g=[1,2,3,4,6],c=5,o="collapse").to_dataframe(names=selected_cols+["distance_to_tss","gene"])
    #mutations_merged = mutations_merged.rename(columns=dict(zip(mutations_merged.columns,target_columns)))
    shape_before = df.shape
    mutations_merged = pd.merge(df,mutations_merged,how="left",on=selected_cols)
    shape_after = mutations_merged.shape

    ### manage ties by picking the one near pancan
    print(shape_before, shape_after)
    assert shape_before[0] == shape_after[0]
    assert shape_before[1] + 2 == shape_after[1]
    mutations_merged["distance_to_tss"] = mutations_merged["distance_to_tss"].apply(lambda x: None if x==-1 else x)

    return mutations_merged


def compute_closest_peak_annotation(df,peak_df):

    #df: needs to have "Chromosome","hg38_start","hg38_end","mutation_id"
    #peak_df: needs to have seqnames, start, end
    ### for each mutation, compute and annotate the closest peak summit

    peak_df_summit = peak_df.copy()
    peak_df_summit["summit"] = (peak_df_summit["start"] + peak_df_summit["end"])//2
    peak_df_summit["summit_1"] = peak_df_summit["summit"] + 1
    peak_df_summit = peak_df_summit[["seqnames","summit","summit_1","start","end"]]
    peak_bed = BedTool.from_dataframe(peak_df_summit)
    
    selected_cols = ["Chromosome","hg38_start","hg38_end","mutation_id"]
    mutations_bed = BedTool.from_dataframe(df[selected_cols]) ## has 4 columns

    target_columns = selected_cols + ["peak_chr","peak_summit","peak_summit_1","peak_start","peak_end"] + ["distance_to_summit"] 
    mutations_merged = mutations_bed.sort().closest(peak_bed.sort(), d=True, t="first").to_dataframe(names=target_columns).drop(["peak_summit","peak_summit_1"],axis=1) ### has 9 columns
    #mutations_merged = mutations_merged.rename(columns=dict(zip(mutations_merged.columns,target_columns)))
    shape_before = df.shape
    mutations_merged = pd.merge(df,mutations_merged,how="left",on=selected_cols)
    shape_after = mutations_merged.shape

    ### manage ties by picking the one near pancan
    print(shape_before, shape_after)
    assert shape_before[0] == shape_after[0]
    assert shape_before[1] + 4 == shape_after[1]
    mutations_merged["distance_to_summit"] = mutations_merged["distance_to_summit"].apply(lambda x: None if x==-1 else x)

    return mutations_merged


def ingene_indicator(gene_string,genelist):
    ### gene_string: a comma separated string list of gene IDs
    ### genelist: a list of gene IDs 
    ## return 1 if any of the 'genes' in gene_sting exists in genelist
    genenames = gene_string.split(",")
    for g in genenames:
        if g in genelist:
            return 1
    else:
        return 0


def get_kbp_tss_stats(tss_df,peaks_df_dict,peak_subsetter_fn,kbps=[1,2,5,10,20,100,500,1000]):
    kbp_to_tss_stats = {}
    for kbp in kbps:
        print(kbp)
        tss_bed = BedTool.from_dataframe(tss_df).slop(b=kbp*1000,g=sizesfile)
        all_tcga_cancers = list(peaks_df_dict.keys())
        for cancer_name in all_tcga_cancers:
            print(cancer_name)
            peak_df = peak_subsetter_fn(peaks_df_dict[cancer_name])
            peak_bed = BedTool.from_dataframe(peak_df)
            tss_bed = tss_bed.intersect(peak_bed,c=True)
        tss_bed = tss_bed.to_dataframe(names=list(tss_df.columns) + all_tcga_cancers)
        kbp_to_tss_stats[kbp] = tss_bed
    return kbp_to_tss_stats


def compute_intermutational_dist(df):

    ### sort the df by chr, start
    ### compute distance between each mutation..

    closest_distances = []
    for chm in df["Chromosome"].unique():
        df_chm = df[df["Chromosome"]==chm].sort_values(["hg38_start"])

        positions_sorted = df_chm["hg38_start"].tolist()
        positions_minus1 = np.array([-1e20] + positions_sorted[:-1])
        positions_plus1 = np.array(positions_sorted[1:] + [1e20])
        positions_sorted = np.array(positions_sorted)

        prev_dist = positions_sorted - positions_minus1
        next_dist = positions_plus1 - positions_sorted

        closest_distances += list(np.minimum(prev_dist,next_dist))

    return closest_distances

def compute_vierstra_groups(df,cleaned_motifs_file,colname_begin):

    cleaned_motifs = pd.read_csv(cleaned_motifs_file)
    motif_cols = ["seqnames","start","end","group_name"]
    cleaned_bed= BedTool.from_dataframe(cleaned_motifs[motif_cols]) ## 4 columns
    
    mutations_cols = ["Chromosome","hg38_start","hg38_end","mutation_id"] 
    mutations_bed = BedTool.from_dataframe(df[mutations_cols].sort_values(mutations_cols)) ## 4 columns
    
    target_columns = mutations_cols + [f"{colname_begin}_{x}" for x in motif_cols] + [f"{colname_begin}_motif_hits"]
    mutations_merged = mutations_bed.intersect(cleaned_bed,wao=True).to_dataframe(names=target_columns) ### 9 columns

    mutations_merged = mutations_merged.drop([f"{colname_begin}_seqnames", \
                                              f"{colname_begin}_start", \
                                              f"{colname_begin}_end",\
                                              f"{colname_begin}_motif_hits"], \
                                              axis=1).sort_values(mutations_cols) ## has 5 columns
    mutations_merged = BedTool.from_dataframe(mutations_merged).groupby(g=[1,2,3,4],c=5,o="collapse").to_dataframe(names=mutations_cols+[f"{colname_begin}_groups"])
    #mutations_merged = mutations_merged.rename(columns=dict(zip(mutations_merged.columns,target_columns)))
    shape_before = df.shape
    mutations_merged = pd.merge(df,mutations_merged,how="left",on=mutations_cols)
    shape_after = mutations_merged.shape

    ### manage ties by picking the one near pancan
    print(shape_before, shape_after)
    assert shape_before[0] == shape_after[0]
    assert shape_before[1] + 1 == shape_after[1]

    return mutations_merged


def search_names_in_vierstra_group(vierstra_group,names_to_search):
    assert type(names_to_search) == list
    for name in names_to_search:
        if name in vierstra_group:
            return 1
    return 0

def get_motif_enrichment(source_df,motif_names,thresh):
    
    df = source_df.copy()
    df["motif_indicator"] = df["vierstra_groups"].apply(lambda x : search_names_in_vierstra_group(x,motif_names))
    print(df["motif_indicator"].value_counts())
    
    colname = "diff_summit_centered"
    print(thresh)
    l1 = df[(df[colname]>thresh) & (df["motif_indicator"]==1)]
    l2 = df[(df[colname]>thresh) & (df["motif_indicator"]==0)]
    l3 = df[(df[colname]<=thresh) & (df["motif_indicator"]==1)]
    l4 = df[(df[colname]<=thresh) & (df["motif_indicator"]==0)]

    fisher_matrix = np.array([[len(l1),len(l2)],[len(l3),len(l4)]])
    fold_change,pval = scipy.stats.fisher_exact(fisher_matrix)
    return fisher_matrix,fold_change,pval