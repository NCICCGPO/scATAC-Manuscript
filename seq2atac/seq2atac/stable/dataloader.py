import numpy as np
from pyfaidx import Fasta
import pandas as pd
import tensorflow as tf
import math
import pyBigWig
from tqdm import tqdm

IN_MAP = np.asarray([[0,0,0,0],
                     [1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,1]], dtype='float')


def get_base_encoding():
    base_encoding = np.zeros((ord('Z') + 1, 1), dtype=int)
    base_encoding[ord('A')] = 1
    base_encoding[ord('C')] = 2
    base_encoding[ord('G')] = 3
    base_encoding[ord('T')] = 4
    return base_encoding


def one_hot_encode(seqs, revcomp=False):
    base_encoding = get_base_encoding()
    if isinstance(seqs, str):# or isinstance(seqs, unicode):
        seqs = [seqs]
    int_seqs = np.concatenate([base_encoding[np.frombuffer(str(s).encode(),dtype=np.byte,count=len(s))] for s in seqs],axis=1).T
    out = IN_MAP[int_seqs]
    return out

def bed_to_numpy(df,fasta_seq):
    assert df.shape[1] >= 3
    xtrain=[]
    for idx,row in df.iterrows():
        chm = str(row[0])
        start = int(row[1])
        end = int(row[2])
        forward_seq=fasta_seq[chm][start:end]
        xtrain.append(forward_seq)
    xtrain=one_hot_encode(xtrain)
    return xtrain


def apply_jitter_augment(master_df,jitter_multiplicity=5,jitter_len=10):
    # Jitter augmentation
    jittered_peak_dfs = []
    choices = [num for num in range(-jitter_len,jitter_len+1) if num!=0]
    for i in range(jitter_multiplicity):

        temp = master_df.copy(deep=True)
        shifts = np.random.choice(choices,size=len(temp),replace=True)
        temp["peak_start"] += shifts
        temp["peak_end"] += shifts
        temp["neg_start"] += shifts
        temp["neg_end"] += shifts
        
        jittered_peak_dfs.append(temp)

    return pd.concat(jittered_peak_dfs+[master_df],axis=0,ignore_index=True)

def jitter_peaks(df,jitter_multiplicity=5,jitter_len=10):
    # Jitter augmentation
    jittered_peak_dfs = []
    choices = [num for num in range(-jitter_len,jitter_len+1) if num!=0]
    for i in range(jitter_multiplicity):

        temp = df.copy(deep=True)
        shifts = np.random.choice(choices,size=len(temp),replace=True)
        temp.iloc[:,1] += shifts
        temp.iloc[:,2] += shifts
        
        jittered_peak_dfs.append(temp)

    return pd.concat(jittered_peak_dfs,axis=0,ignore_index=True)


def compute_signal(df,signal_file):
    pos_phylops = []
    with pyBigWig.open(signal_file,"r") as bw:
        for idx,row in tqdm(df.iterrows(), total=len(df)):
            chm,start,end = row[:3]
            pos_phylops.append(np.nan_to_num(bw.values(chm,start,end,numpy=True)))
    return np.array(pos_phylops)


class DataLoaderMiniBatchMatched(tf.keras.utils.Sequence):
    
    def __init__(self,master_df,batch_size,fasta_seq):

        required_cols = ["peak_chr","peak_start","peak_end","peak_label",
                         "neg_chr","neg_start","neg_end","neg_label"]
        for col in required_cols:
            assert col in master_df.columns, "Wrong master_df format"

        ### effective batch size = batch_size * 2 * 4 (2 for negatives, 4 for jitter)
        self.batch_size = batch_size 
        self.fasta_seq = fasta_seq

        self.master_df = master_df

        
    def __len__(self):
        return math.ceil(self.master_df.shape[0] / float(self.batch_size))
    
    def __getitem__(self,idx):
        
        # get some peaks
        master_subset = self.master_df[(idx*self.batch_size):(idx*self.batch_size)+(self.batch_size)]
        
        ## compute sequence and label
        X_train_pos=bed_to_numpy(master_subset[["peak_chr","peak_start","peak_end"]],self.fasta_seq)
        y_train_pos=np.ones((len(X_train_pos)))

        X_train_neg=bed_to_numpy(master_subset[["neg_chr","neg_start","neg_end"]],self.fasta_seq)
        y_train_neg=np.zeros((len(X_train_neg)))

        X_train_batch = np.concatenate([X_train_pos, X_train_neg])
        y_train = np.concatenate([y_train_pos, y_train_neg])

        return ({'sequence':X_train_batch} ,{'dnase':y_train})
    
    def on_epoch_end(self):
        # shuffle dataframe
        self.master_df=self.master_df.sample(frac=1).reset_index(drop=True)

