import pandas as pd
import glob
from tqdm import tqdm

from pyfaidx import Fasta
from pybedtools import BedTool
from liftover import get_lifter

from seq2atac.analysis import gnomad_file, fasta_file, rmsk_file, phylop_file, sizesfile
from seq2atac.stable import compute_gc_bed, compute_signal, read_pickle
from seq2atac.analysis.mutation_utils import get_mutation_type
from seq2atac.analysis import get_promoterai_tss
import numpy as np

"""
Some notes

- df: has to have certain column names
- filter* and annotate* functions usually cannot handle duplicates reliably, so pls check shapes before and after


"""


def cleanGnomadVariants():

    """
    Filters GNOMAD v3 file to produce a variant set

    Inputs:
        af_thresh: 

    """
    ## gnomad_file: taken from deep_learning common resources
    ### its format is a gzipped tab separated file with chrom, pos, ref, alt positions
    ## af_thresh: 

    ## fixes "pos" (subtracts 1)
    ## fixes "ref" and "alt" by forcing them to be capitals
    ## removes rows with ref/alt not in [A,C,G,T]
    ## removes rows where fasta[chrom][pos] != ref
    ## deduplicates on [chrom, pos, alt]

    fasta_seq = Fasta(fasta_file)

    all_rows = []
    for chunk in pd.read_csv(gnomad_file, sep="\t",chunksize=1e6, compression="gzip"):
        
        # ### subset to AF
        # chunk = chunk[chunk["gnomad_af"] >= af_thresh]
        
        ### Fix start
        chunk["pos"] = chunk["pos"] - 1
        
        ### Remove non-SNPs
        chunk["ref"] = chunk["ref"].apply(lambda x:x.upper())
        chunk["alt"] = chunk["alt"].apply(lambda x:x.upper())
        vocab = ["A","G","C","T"]
        chunk = chunk[chunk["ref"].isin(vocab)]
        chunk = chunk[chunk["alt"].isin(vocab)]
        
        ### Verify ref and remove those that are wrong
        correct_ref_indicator = []
        for idx,row in chunk.iterrows():
            chrom,pos,ref = row["chrom"], int(row["pos"]), row["ref"]

            if fasta_seq[chrom][pos] != ref:
                correct_ref_indicator.append(0)
            else:
                correct_ref_indicator.append(1)
        chunk["correct_ref_indicator"] = correct_ref_indicator
        chunk = chunk[chunk["correct_ref_indicator"] == 1].drop("correct_ref_indicator",axis=1)
        
        all_rows.append(chunk)
        print(len(all_rows))
        
    gnomad_af_df = pd.concat(all_rows,axis=0,ignore_index=True).drop_duplicates(["chrom","pos","alt"])
    # gnomad_af_df["mutation_id"] = [f"C{idx}" for idx in gnomad_af_df.index]
    return gnomad_af_df


def hg19_to_hg38(df_mutations):

    fasta_seq = Fasta(fasta_file)
    converter = get_lifter('hg19', 'hg38')
    # all mutations is a file with a df with Chromosome, Start_position in hg19, Reference_Allele, Tumor_Seq_Allele2

    ### 0 indexed position of mutation
    df_mutations['hg19_start']=(df_mutations['Start_position']-1).astype(int)

    ### liftover to hg38
    print("lifting over")
    new_ref_allele=[]
    new_strand_list=[]

    for idx,row in tqdm(df_mutations.iterrows(), total=len(df_mutations)):
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
        chrom,pos = row["Chromosome"], int(row["hg38_start"])

        half = 1364//2
        left = int(pos - half)
        right = int(pos + half)

        seq1 = str(fasta_seq[chrom][left:right])

        if len(seq1) != 1364:
            flankable_indicator.append(0)
        else:
            flankable_indicator.append(1)
    df_mutations["flankable_indicator"] = flankable_indicator

    df_mutations = df_mutations[df_mutations["flankable_indicator"] == 1].reset_index(drop=True)
    print(f"Removed those that cant be flanked to 1364")
    df_mutations = df_mutations.drop("flankable_indicator",axis=1)
    print(df_mutations.shape)

    df_mutations = df_mutations.drop(["Start_position","End_position","Strand","hg19_start"],axis=1)
    return df_mutations



def filterannotatePeaks(df, peak_bedfile, annotate_only):

    """
    Inputs:
        df: csv, 
        peakfile: csv => seqnames, start, end, peak_index
    
    """

    selected_cols = ["Chromosome","hg38_start","hg38_end","mutation_id"]
    df_bed = BedTool.from_dataframe(df[selected_cols]) # 4 cols
    peak_bed = BedTool(peak_bedfile) # 4 cols
    total_len = len(df)

    df_intersecting_peaks = None
    if annotate_only:
        df_intersecting_peaks = df_bed

    else:
        ### filter to overlapping "peak_bedfile"
        df_intersecting_peaks = df_bed.intersect(peak_bed,u=True) # 4 cols
    
    intersecting_len = len(df_intersecting_peaks)

    ### now annotate the closest feature in "peak_bedfile" (make sure to sort)
    peak_df = pd.read_csv(peak_bedfile,sep="\t",header=None)
    peak_df.columns = ['peak_chr','peak_start','peak_end','peak_index']
    peak_df['peak_summit'] = (peak_df['peak_start'] + peak_df['peak_end'])//2
    peak_df['peak_summit_1'] = peak_df['peak_summit'] + 1
    peak_df_colnames = ['peak_chr','peak_summit','peak_summit_1','peak_start','peak_end','peak_index']
    peak_summits_bed = BedTool.from_dataframe(peak_df[peak_df_colnames])

    df_intersecting_peaks_closest = df_intersecting_peaks.sort().closest(peak_summits_bed.sort(), d=True, t="first")  # 4 + 7 cols
    target_names = selected_cols + peak_df_colnames + ["distance_to_summit"]
    df_intersecting_peaks_closest = df_intersecting_peaks_closest.to_dataframe(names=target_names).drop('peak_summit_1',axis=1)
    
    assert len(df_intersecting_peaks_closest) == intersecting_len


    df_merged = df_intersecting_peaks_closest.merge(df,on=selected_cols,how="left")

    assert len(df_merged) == intersecting_len

    return df_merged


def filterRepeatMask(df):

    df_bed = BedTool.from_dataframe(df)
    repeat_mask_bed = BedTool(rmsk_file)
    
    df_colnames = list(df.columns)
    df_intersect = df_bed.intersect(repeat_mask_bed, v=True).to_dataframe(names=df_colnames)
    return df_intersect

def filter4Callers(df):

    assert "i_NumCallers" in df.columns
    return df[df["i_NumCallers"]=="4"]

def filtergnomAD(df):
    ### add 2 columns to df
    ## df: mutations format
    ## must have Chromosome, hg38_start, hg38_end, Reference_Allele, Tumor_Seq_Allele2

    gn = pd.read_csv(gnomad_file,sep="\t",compression = "gzip")
    gn["start"] = gn["pos"] - 1
    gn.drop("pos",axis=1,inplace=True)
    gn["common_variant_indicator"] = 1
    print(gn.shape)

    df_mutations = pd.merge(df, gn, \
                    how="left", \
                    left_on=["Chromosome","hg38_start","Tumor_Seq_Allele2"], \
                    right_on=["chrom","start","alt"])
    df_mutations.drop(["chrom","start","end","ref","alt"],axis=1,inplace=True)
    print("Afer merging with gnomAD: ", df_mutations.shape)

    df_mutations["common_variant_indicator"] = df_mutations["common_variant_indicator"].fillna(0)
    print(df_mutations.shape)
    return df_mutations[df_mutations["common_variant_indicator"]!=1].drop("common_variant_indicator",axis=1)

def filterCommonVariants(df, common_variants_csv):

    gn = pd.read_csv(common_variants_csv)[["Chromosome","hg38_start","Tumor_Seq_Allele2","status","gnomad_af"]]
    gn["common_variant_indicator"] = 1
    print(gn.shape)

    df_mutations = pd.merge(df, \
                            gn, \
                            how="left", \
                            on=["Chromosome","hg38_start","Tumor_Seq_Allele2"])
    print(df_mutations.shape)

    # df_mutations["status"] = df_mutations["status"].fillna("ABSENT")
    df_mutations["gnomad_af"] = df_mutations["gnomad_af"].fillna(-1)
    df_mutations["common_variant_indicator"] = df_mutations["common_variant_indicator"].fillna(0)
    print(df_mutations.shape)

    # df_mutations = df_mutations[df_mutations["status"]=="ABSENT"]
    # df_mutations.to_csv(f"{outdir}/{cancer_name}_mutation_scores.tsv",sep="\t",index=None)
    return df_mutations[df_mutations["common_variant_indicator"]!=1].drop("common_variant_indicator",axis=1)


# def annotateScore(df,score_files_pattern):

#     fold_score_files = glob.glob(score_files_pattern)
#     assert len(fold_score_files) == 5

#     df_scored = df.copy()
#     for fidx,f in enumerate(fold_score_files):
#         proba_df = read_pickle(f)
#         assert not proba_df.isnull().values.any()
#         assert len(proba_df) == len(df)
#         proba_df = proba_df.rename(columns={"proba_ref": f"proba_ref_{fidx}", \
#                                             "proba_alt": f"proba_alt_{fidx}"})
#         df_scored = df_scored.merge(proba_df, how="left")

#     df_scored["proba_ref"] = df_scored[[f"proba_ref_{fidx}" for fidx in range(5)]].mean(axis=1)
#     df_scored["proba_alt"] = df_scored[[f"proba_alt_{fidx}" for fidx in range(5)]].mean(axis=1)

#     assert not df_scored[["proba_ref","proba_alt"]].isnull().values.any()

#     df_scored = df_scored.drop([f"proba_ref_{fidx}" for fidx in range(5)]+[f"proba_alt_{fidx}" for fidx in range(5)], axis=1)
    
#     return df_scored


def annotateScore(df,score_files_pattern,remove_foldwise=True):

    fold_score_files = glob.glob(score_files_pattern)
    assert len(fold_score_files) == 5

    merge_on = ["Chromosome","hg38_start","hg38_end","Reference_Allele","Tumor_Seq_Allele2","mutation_id"]

    df_scored = df.copy()
    for fidx,f in enumerate(fold_score_files):
        proba_df = read_pickle(f)
        assert not proba_df.isnull().values.any()
        proba_df = proba_df.rename(columns={"proba_ref": f"proba_ref_{fidx}", \
                                            "proba_alt": f"proba_alt_{fidx}"})
        df_scored = df_scored.merge(proba_df, on=merge_on, how="left")

    df_scored["proba_ref"] = df_scored[[f"proba_ref_{fidx}" for fidx in range(5)]].mean(axis=1)
    df_scored["proba_alt"] = df_scored[[f"proba_alt_{fidx}" for fidx in range(5)]].mean(axis=1)

    assert not df_scored[["proba_ref","proba_alt"]].isnull().values.any()

    if remove_foldwise:
        df_scored = df_scored.drop([f"proba_ref_{fidx}" for fidx in range(5)]+[f"proba_alt_{fidx}" for fidx in range(5)], axis=1)
    
    return df_scored


def annotateMotif(df,motif_file,slop,colname_begin):

    cleaned_motifs = pd.read_csv(motif_file)
    motif_cols = ["seqnames","start","end","group_name"]
    cleaned_bed= BedTool.from_dataframe(cleaned_motifs[motif_cols]).slop(b=slop,g=sizesfile) ## 4 columns
    
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
    shape_before = df.shape
    mutations_merged = pd.merge(df,mutations_merged,how="left",on=mutations_cols)
    shape_after = mutations_merged.shape

    ### manage ties by picking the one near pancan
    print(shape_before, shape_after)
    assert shape_before[0] == shape_after[0]
    assert shape_before[1] + 1 == shape_after[1]

    return mutations_merged

def annotateBedfile(df, bedfile, colname):

    """
    Inputs:
        df: csv, 
        peakfile: csv => seqnames, start, end, peak_index
    
    """

    colnames = ["Chromosome","hg38_start","hg38_end","Tumor_Seq_Allele2","mutation_id"]
    df_bed = BedTool.from_dataframe(df[colnames])
    
    
    df_annotated = df_bed.intersect(BedTool(bedfile),c=True).to_dataframe(names=colnames+[colname])
    len_before = len(df_annotated)
    df_annotated = df_annotated.merge(df,how="left")
    
    assert len(df_annotated) == len_before
    
    df_annotated[colname] = df_annotated[colname].apply(lambda x : int(x > 0))

    return df_annotated


def annotateNearestGene(df,genelist):

    ## df: a bed format dataframe:  must have "Chromosome","hg38_start","hg38_end","mutation_id"
    ## genelist: list of genes of interest
    ### for each mutation, add 2 columns - gene, distance_to_tss

    ###  Get TSS information
    tss_df = get_promoterai_tss()
    tss_df_genelist = tss_df[tss_df["gene"].isin(genelist)]
    tss_bed = BedTool.from_dataframe(tss_df_genelist)
    
    ### Create bedfile from df
    selected_cols = ["Chromosome","hg38_start","hg38_end","mutation_id"]
    mutations_bed = BedTool.from_dataframe(df[selected_cols]) ## has 4 columns
    target_columns = selected_cols + ["gene_chr","gene_start","gene_end","gene"] + ["distance_to_tss"]

    ### Annotate closest (required to sort both inputs) 
    mutations_merged = mutations_bed.sort().closest(tss_bed.sort(), d=True).to_dataframe(names=target_columns) ### has 9 columns
    ### If ties, just make the annotation comma separated
    mutations_merged = mutations_merged.drop(["gene_chr","gene_start","gene_end"],axis=1).sort_values(selected_cols) ## has 6 columns
    mutations_merged = BedTool.from_dataframe(mutations_merged).groupby(g=[1,2,3,4,6],c=5,o="collapse").to_dataframe(names=selected_cols+["distance_to_tss","gene"])

    ### Merge this information with other columns
    shape_before = df.shape
    mutations_merged = pd.merge(df,mutations_merged,how="left",on=selected_cols)
    shape_after = mutations_merged.shape
    print(shape_before, shape_after)

    ### Assert every mutation got annotated
    assert shape_before[0] == shape_after[0]
    assert shape_before[1] + 2 == shape_after[1]
    mutations_merged["distance_to_tss"] = mutations_merged["distance_to_tss"].apply(lambda x: None if x==-1 else x)

    return mutations_merged


def annotatePhylopScore(df,colname,window=None):


    if window == None:
        df[colname] = compute_signal(df[["Chromosome","hg38_start","hg38_end"]], phylop_file)

    else:
        df_temp = df[["Chromosome","hg38_start","hg38_end"]].copy()
        df_temp["hg38_start"] -= window//2
        df_temp["hg38_end"] = df_temp["hg38_start"] + window
        df[colname] = np.mean(compute_signal(df_temp[["Chromosome","hg38_start","hg38_end"]], phylop_file), axis=-1)
    return df



def annotatePhylopScorePeak(df,colname,window=None):


    df_temp = df[["peak_chr","peak_start","peak_end"]].copy()
    df_temp["peak_summit"] = (df_temp["peak_start"] + df_temp["peak_end"])//2

    if window == None:
        df_temp["peak_summit_1"] = df_temp["peak_summit"] + 1
        df[colname] = compute_signal(df_temp[["peak_chr","peak_summit","peak_summit_1"]], phylop_file)

    else:
        df_temp["peak_summit"] -= window//2
        df_temp["peak_summit_1"] = df_temp["peak_summit"] + window
        df[colname] = np.mean(compute_signal(df_temp[["peak_chr","peak_summit","peak_summit_1"]], phylop_file), axis=-1)
    return df


# def annotateTrinuc(df,colname):
#     lambda_compute_trinuc = lambda x: compute_trinuc(x["Chromosome"],
#                                                  x["hg38_start"],
#                                                  x["hg38_end"],
#                                                  x["Reference_Allele"],
#                                                  x["Tumor_Seq_Allele2"])

#     df[colname] = df.apply(lambda_compute_trinuc, axis=1) 
#     return df


def annotateGC1364(df,colname):

    fasta_seq = Fasta(fasta_file)
    df[colname] = compute_gc_bed(df[["Chromosome","hg38_start","hg38_end"]], fasta_seq, 1364, roundoff=2)
    return df

def annotateGCPeak(df,colname):

    fasta_seq = Fasta(fasta_file)
    df[colname] = compute_gc_bed(df[["peak_chr","peak_start","peak_end"]], fasta_seq, 1364, roundoff=2)
    return df


def computeTrinuc(df,fasta_seq):
        
    trinucs = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        chm, start, ref, alt = row["Chromosome"],row["hg38_start"],row["Reference_Allele"],row["Tumor_Seq_Allele2"]
        predicted_ref = str(fasta_seq[chm][start])
        assert predicted_ref.upper() == ref.upper()
        left = str(fasta_seq[chm][start-1])
        right = str(fasta_seq[chm][start+1])
        mid = get_mutation_type(ref,alt)
        trinuc = left + mid + right
        trinuc = trinuc.upper()
        trinucs.append(trinuc)

    return trinucs
