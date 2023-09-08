from seq2atac.stable import bed_to_numpy
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
def compute_auroc_auprc_matched(model,val_df,fasta_seq):
    X_val_positives = bed_to_numpy(val_df[["peak_chr","peak_start","peak_end"]],fasta_seq)
    y_val_positives = np.ones((len(X_val_positives)))
    y_pred_positives = model.predict(X_val_positives,verbose=1).ravel()

    X_val_negatives = bed_to_numpy(val_df[["neg_chr","neg_start","neg_end"]],fasta_seq)
    y_val_negatives = np.zeros((len(X_val_negatives)))
    y_pred_negatives = model.predict(X_val_negatives,verbose=1).ravel()

    y_val = np.concatenate([y_val_positives,y_val_negatives])
    y_pred = np.concatenate([y_pred_positives,y_pred_negatives])

    # AuROC
    auroc = roc_auc_score(y_val,y_pred)
    # AuPRC
    auprc = average_precision_score(y_val,y_pred)

    return auroc,auprc

def predict_df(model,master_df,fasta_seq):

    X_val_positives = bed_to_numpy(master_df[["peak_chr","peak_start","peak_end"]],fasta_seq)
    y_pred_positives = model.predict(X_val_positives,verbose=1).ravel()

    X_val_negatives = bed_to_numpy(master_df[["neg_chr","neg_start","neg_end"]],fasta_seq)
    y_pred_negatives = model.predict(X_val_negatives,verbose=1).ravel()

    df = pd.DataFrame()
    df.loc[:,"peak_pred"] = y_pred_positives
    df.loc[:,"neg_pred"] = y_pred_negatives

    return df

def predict_bed(model,peak_df,fasta_seq):

    X_val = bed_to_numpy(peak_df,fasta_seq)
    y_pred_positives = model.predict(X_val,verbose=1).ravel()

    return y_pred_positives

