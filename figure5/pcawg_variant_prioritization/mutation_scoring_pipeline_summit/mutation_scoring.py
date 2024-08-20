import sys
import pandas as pd
from seq2atac.stable import one_hot_encode, write_pickle, read_pickle
from seq2atac.stable.models import model_name_to_fn
from pybedtools import BedTool
from pyfaidx import Fasta
import numpy as np
from tqdm import tqdm
from copy import deepcopy

fasta_file = './fasta_hg38.fa' ## TODO: add fasta file path

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



def main(mutations_file,model_type,model_file,outfile):

    print("Arguments")
    print("mutations_file: ", mutations_file)
    print("model_type: ", model_type)
    print("model_file: ",model_file)
    print("outfile: ",outfile)

    ### get fasta file
    fasta_seq=Fasta(fasta_file)

    ### load model
    model = model_name_to_fn[model_type]()
    model.load_weights(model_file)

    print("Reading mutations")
    df_mutations = None
    try:
        df_mutations = pd.read_csv(mutations_file,sep=",")
    except:
        df_mutations = read_pickle(mutations_file)
    assert len(df_mutations), "incorrect file format"
    df_mutations["peak_start"] = df_mutations["peak_start"].astype(int)
    df_mutations["peak_end"] = df_mutations["peak_end"].astype(int)

    #df_preds = pd.DataFrame(index=df_mutations.index)
    df_preds = df_mutations[["Chromosome","hg38_start","hg38_end","Reference_Allele","Tumor_Seq_Allele2","mutation_id"]].copy()
    index_array = np.array(df_preds.index)
    num_batches = len(df_preds)//64000

    if num_batches <= 1:
        num_batches = 1

    ref_phylops = []
    alt_phylops = []

    for X_batch in tqdm(np.array_split(index_array,num_batches)):
        ref_seqs, alt_seqs = get_alt_sequence(df_mutations.loc[X_batch],1364,fasta_seq)
        X_ref = one_hot_encode(ref_seqs)
        X_alt = one_hot_encode(alt_seqs)

        pred_ref = model.predict(X_ref,batch_size=128)
        pred_alt = model.predict(X_alt,batch_size=128)

        df_preds.loc[X_batch,f"proba_ref"] = pred_ref.ravel()
        df_preds.loc[X_batch,f"proba_alt"] = pred_alt.ravel()

    # df_preds.to_csv(outfile)
    write_pickle(df_preds,outfile)

import sys
if __name__=="__main__":
    mutations_file,model_type,model_file,outfile = sys.argv[1:]
    main(mutations_file,model_type,model_file,outfile)
