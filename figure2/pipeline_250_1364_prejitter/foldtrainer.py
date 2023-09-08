import sys
import pandas as pd
from seq2atac.stable import create_splits, hg38_splits, train_classifier, compute_auroc_auprc_matched, predict_df, jitter_peaks, compute_gc_bed, bed_to_numpy
from seq2atac.stable.models.convolutional import get_bpnet_model
from seq2atac.stable.models.transformers import get_tx_model
from seq2atac.stable.negative_sampling import create_matched_master
from seq2atac.stable.dataloader import DataLoaderMiniBatchMatched
from pyfaidx import Fasta
import os

import numpy as np
import scipy.stats
from pyfaidx import Fasta
from pybedtools import BedTool
import tensorflow as tf

model_name_to_fn = {"conv_1364": lambda : get_bpnet_model(1364,8),
                    "tx_1364": lambda: get_tx_model(1364,8,2)}

fold_number_to_seed = [94404,94305,94086,95070,600028,575025]

def main(peakfile,outdir,fold_number,gc_match_size,model_input_size,model_type):

    print("Arguments")
    print("peakfile: ",peakfile)
    print("outdir: ", outdir)
    print("fold_number: ", fold_number)
    print("gc match size: ",gc_match_size)
    print("model input size: ",model_input_size)
    print("model type: ",model_type)

    jitter_len = 10
    jitter_multiplicity = 5
    print("Jitter params: ")

    seed_value = fold_number_to_seed[fold_number]
    np.random.seed(seed_value)
    print("Seed set to: ",seed_value)

    ### outdir is the fold dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print("Created: ",outdir)

    ### get fasta file
    fasta_file = '/illumina/scratch/deep_learning/lsundaram/singlecelldatasets/TCGA/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta'
    fasta_seq=Fasta(fasta_file)

    ### master windowed gc file
    windowed_1364_gc_file = f"/illumina/scratch/deep_learning/akumar22/generated_data/windowed_genome_gc_{gc_match_size}_{model_input_size}.tsv"

    ### read standard peak file processed by Laksshman (NOTE)
    print("Reading peak file")
    peak_csv_df = pd.read_csv(peakfile)[["seqnames","start","end"]]
    ### jitter the training peaks
    jittered_train_peaks = jitter_peaks(peak_csv_df[peak_csv_df["seqnames"].isin(hg38_splits[fold_number]["train"])],
                                        jitter_multiplicity=jitter_multiplicity,
                                        jitter_len=jitter_len)
    peak_csv_df = pd.concat([peak_csv_df, jittered_train_peaks],axis=0,ignore_index=True).sort_values(["seqnames","start"]).reset_index(drop=True)

    ## compute gc
    peak_csv_df["gcs"] = compute_gc_bed(peak_csv_df,fasta_seq,gc_match_size,roundoff=2)

    ## move to a bed file
    positives_file=f"{outdir}/positives.bed"
    peaks_bed = BedTool.from_dataframe(peak_csv_df).moveto(positives_file)

    print("Complementing with positives...")
    windowed_gc_bed = BedTool(windowed_1364_gc_file)
    nopeaks_file=f"{outdir}/nopeaks.bed"
    nopeaks_bed = windowed_gc_bed.intersect(peaks_bed,wa=True,v=True).moveto(nopeaks_file)
    print("Written to: ",nopeaks_file)

    print("Creating matched master file")
    master_df_path = f"{outdir}/master.csv"
    master_df = create_matched_master(positives_file, nopeaks_file, model_input_size, master_df_path)
    ## create train, val, test sets

    ### assert p value
    gc_pos = compute_gc_bed(master_df[["peak_chr","peak_start","peak_end"]], fasta_seq, gc_match_size, roundoff=None)
    gc_neg = compute_gc_bed(master_df[["neg_chr","neg_start","neg_end"]], fasta_seq, gc_match_size, roundoff=None)
    assert np.all(np.isclose(np.array(gc_pos),np.array(gc_neg),atol=0.01)), "Not matched properly"
    ranksum, pvalue = scipy.stats.ranksums(gc_pos, gc_neg)
    print("GC matching p-value: ", pvalue)
    if pvalue < 0.05:
        print("Bad matching!!! Repeat experiment")
        return 0.0

    if fold_number != 0:
        try:
            os.remove(nopeaks_file)
        except:
            print("Temp files could not be removed")

    train_df, val_df, test_df = create_splits(master_df,
                                        train_split=hg38_splits[fold_number]["train"],
                                        val_split=hg38_splits[fold_number]["valid"],
                                        test_split=hg38_splits[fold_number]["test"],
                                        chm_colname="peak_chr")

    print("Building dataloader")
    b_size = 256
    train_gen = DataLoaderMiniBatchMatched(train_df,b_size,fasta_seq)
    val_gen = DataLoaderMiniBatchMatched(val_df,b_size,fasta_seq)

    print("Building model...")
    model = model_name_to_fn[model_type]()
    ### Get data generators

    model_path = outdir + "/model.h5"
    model = train_classifier(model, train_gen, val_gen, model_path)

    print("Analysis")
    auroc,auprc = compute_auroc_auprc_matched(model,val_df,fasta_seq)
    print("AUROC: ",auroc)
    print("AUPRC: ",auprc)

    if auroc >= 0.75:
        # print("Computing master fold 0 predictions")
        # master_file_fold0 = f"{outdir}/../fold_0/master.csv"
        # master_df_fold0 = pd.read_csv(master_file_fold0)
        # predictions_df = predict_df(model,master_df_fold0,fasta_seq)
        # predictions_df.columns = [f"peak_pred_{fold_number}", f"neg_pred_{fold_number}"]
        # predictions_df.to_csv(outdir+"/fold0_preds.csv",index=None)

        print("Scoring all peaks...")
        peak_df = pd.read_csv(peakfile)[["seqnames","start","end"]]
        peaksize=500
        flank = (model_input_size-peaksize)//2
        peak_df["start"] = peak_df["start"] - flank
        peak_df["end"] = peak_df["start"] + model_input_size
            
        X_val = bed_to_numpy(peak_df[["seqnames","start","end"]],fasta_seq)
        model.load_weights(model_path)
        y_pred_positives = model.predict(X_val,verbose=1).ravel()
        peak_df[f"preds_{fold_number}"] = y_pred_positives

        for gc_num in [150,250,500,1364]:
            peak_df[f"gc_{gc_num}"] = compute_gc_bed(peak_df[["seqnames","start","end"]], fasta_seq, gc_num, roundoff=None)
        peak_df.to_csv(outdir+"/peak_preds.csv",index=None)  
    return auroc

import sys
if __name__=="__main__":
    peakfile = sys.argv[1]
    outdir = sys.argv[2]
    fold_number = int(sys.argv[3])
    gc_match_size = int(sys.argv[4])
    model_input_size = int(sys.argv[5])
    model_type = sys.argv[6]

    auroc = 0.0
    while auroc < 0.75:
        print("Repeating: auroc = ",auroc)
        auroc = main(peakfile,outdir,fold_number,gc_match_size,model_input_size,model_type)
    
    print("Done")


