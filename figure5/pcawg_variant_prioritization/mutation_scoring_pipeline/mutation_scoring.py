import sys
import pandas as pd
from seq2atac.stable import one_hot_encode, write_pickle, read_pickle
from seq2atac.stable.models import model_name_to_fn
from seq2atac.analysis.enrichment_utils import get_refalt_sequence
from pyfaidx import Fasta

import numpy as np
from pyfaidx import Fasta
from tqdm import tqdm
from copy import deepcopy

fasta_file = './fasta_hg38.fa' ## TODO: add fasta file path

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
    # df_mutations = pd.read_csv(mutations_file,sep=",")
    df_mutations = None
    try:
        df_mutations = pd.read_csv(mutations_file,sep=",")
    except:
        df_mutations = read_pickle(mutations_file)
    assert len(df_mutations), "incorrect file format"

    #df_preds = pd.DataFrame(index=df_mutations.index)
    df_preds = df_mutations[["Chromosome","hg38_start","hg38_end","Reference_Allele","Tumor_Seq_Allele2","mutation_id"]].copy()
    index_array = np.array(df_preds.index)
    num_batches = len(df_preds)//64000

    if num_batches <= 1:
        num_batches = 1
    
    df_preds["proba_ref"] = ""
    df_preds["proba_alt"] = ""

    for X_batch in tqdm(np.array_split(index_array,num_batches)):
        ref_seqs, alt_seqs = get_refalt_sequence(df_mutations.loc[X_batch],1364,fasta_seq)
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
