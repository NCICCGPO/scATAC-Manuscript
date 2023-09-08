import sys
from seq2atac.analysis.mutation_processing_pipeline_utils import filterannotatePeaks, filterRepeatMask, filtergnomAD
from seq2atac.analysis.mutation_processing_pipeline_utils import computeTrinuc, annotateNearestGene
from seq2atac.analysis.mutation_utils import ingene_indicator
from seq2atac.stable import read_pickle, write_pickle, compute_gc_bed
from seq2atac.analysis import fasta_file, get_promoterai_tss, get_ogtsg
from pyfaidx import Fasta

### global outdir
def filter_somatic_set(somatic_pkl, peakfile, outfile1, outfile2):

    fasta_seq = Fasta(fasta_file)

    tss_df = get_promoterai_tss()
    all_genes = list(set(tss_df["gene"].tolist()))
    print("num total genes: ",len(all_genes))

    pancan_genes = get_ogtsg()
    print("num pancan genes: ",len(pancan_genes))
    ispancan = lambda x : ingene_indicator(x, pancan_genes)
    
    ### infer the files
    print("reading somatic set from: ",somatic_pkl)
    
    df = read_pickle(somatic_pkl)
    print(df.shape)
    
    ### remove anything but 4 callers
    if "i_NumCallers" in df.columns:
        df = df[df["i_NumCallers"]=="4"]
        print("removed all <4 callers")
    print(df.shape)
    
    ### remove anything marked as repeat
    if "i_repeat_masker" in df.columns:
        df = df[df["i_repeat_masker"].isna()]
        print("removed all repeat masker annotated rows")
    print(df.shape)
    
    ### remove those overlapping `rmsk_file`
    df_rmsk_filtered = filterRepeatMask(df[["Chromosome","hg38_start","hg38_end","mutation_id"]])
    filtered_len = len(df_rmsk_filtered)
    df = df_rmsk_filtered.merge(df)
    assert len(df) == filtered_len
    print("removed those overlapping repeat masker")
    print(df.shape)

    ### remove chrX, chrY
    df = df[(~df["Chromosome"].isin(["chrX","chrY"]))]
    print("removed chr x and y")
    print(df.shape)

    ### remove those overlapping gnomad
    df = filtergnomAD(df)
    print("removed those overlapping gnomad")
    print(df.shape)
    
    ### annotate distance to closest peaks
    shape_before = len(df)
    df = filterannotatePeaks(df, peak_bedfile=peakfile, annotate_only=True)
    assert len(df) == shape_before
    print("After annotating peaks: ",df.shape)

    ### annotate gc and trinuc
    df["gc_mutation"] = compute_gc_bed(df[["Chromosome","hg38_start","hg38_end"]], fasta_seq, 1364, roundoff=None)
    df["gc_peak"] = compute_gc_bed(df[["peak_chr","peak_start","peak_end"]], fasta_seq, 1364, roundoff=None)
    df["trinuc"] = computeTrinuc(df, fasta_seq)

    ### annotate nearest gene
    df = annotateNearestGene(df, all_genes)
    print("After annotating nearest gene: ", df.shape)
    df["closest_pancan"] = df["gene"].apply(ispancan)

    print("writing filtered somatic mutation to: ",outfile1)
    write_pickle(df, outfile1)

    df = df.drop(["peak_chr","peak_start","peak_end","peak_summit","peak_index","distance_to_summit"],axis=1)
    df = filterannotatePeaks(df, peak_bedfile=peakfile, annotate_only=False)
    print("only inside peaks: ",df.shape)
    print("writing filtered somatic mutation to: ", outfile2)
    write_pickle(df, outfile2)

import sys
if __name__=="__main__":
    somatic_pkl, peakfile, outfile1, outfile2 = sys.argv[1:]
    filter_somatic_set(somatic_pkl, peakfile, outfile1, outfile2)
