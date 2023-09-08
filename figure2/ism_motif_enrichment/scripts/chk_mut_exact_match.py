import sys
sys.path.append('/./')
from seq2atac.analysis import fasta_seq
from reg_diffs.scripts import GoF_dev as gof
import numpy as np
import pandas as pd

def subsequence(text, subseq_length):
    if subseq_length <= 0:
        return []
    if subseq_length == 1:
        return list(text)
    text_length = len(text)
    res = []
    tail_length = subseq_length - 1
    for i in range(0, text_length - tail_length):
        for tail in subsequence(text[i+1:], tail_length):
            res.append(text[i] + tail)
    return res

def seq_match(
    query: str, 
    motif: str = 'ACGTACGTACGT',
) -> bool:
    """
    
    Arguments:
      `query`: str
        query is another string 'ACGT'. With mutations, commonly 
        the mutation is in the middle of the query
    """
    
    return motif in subsequence(query, len(motif))

def get_seq(
    fasta_seq,
    mut_midpt: str or None = None,
    n_flank: int = 3, 
    chrom: str = 'chr1',
    start: int = 1,
    end: int or None = None, 
):
    if end is None:
        end = start
        
    # get midpt
    if end > start:
        midpt = ((end - start) // 2) + start
        if (end - start) % 2 == 1:
            midpt = midpt + np.random.choice([0, 1])
    else:
        midpt = start
    
    L = midpt - n_flank
    R = midpt + n_flank + 1
    
    L = str(fasta_seq[chrom][L:midpt])
    if mut_midpt is not None:
        R = mut_midpt + str(fasta_seq[chrom][midpt+1:R])
    else: 
        R = str(fasta_seq[chrom][midpt:R])
        
    return L + R
    
    
def get_mut_df(
    sample: str,
    file: str = './tmp/TCGA/mutations_scoring/gof_experiments/scored_annotated_dataframes/somatic_df_dict_mutation_centered.pkl',
    es_key: str = 'diff_mutation_centered',
    verbose: bool = True,
    
) -> pd.DataFrame :
    """
    
    Returns:
      seq_match: bool
      seq: str
        in case additional queries are required
    """

    # get the files and threshold
    moi = gof.get_somatic_mutations(file=file, es_key=es_key)

    # subset the data
    md = moi.loc[moi['sample']==sample.upper(), :]
    if verbose:
        print('md:', md.shape)
        
    return md

def match_mutVquery(
    alt_seq: str,
    chrom: str,
    loc: int,
    motif: str,
    fasta_seq: object, 
    verbose: bool = False,
) -> bool :
    assert len(alt_seq) == 1, 'only works for single seqs at this pt'
    seq = get_seq(fasta_seq, mut_midpt=alt_seq,
                  n_flank = len(motif) - 1,
                  chrom=chrom, start=loc,)
    if verbose:
        print('query seq:', seq)
        print('motif to search:', motif)
    
    return seq_match(seq, motif=motif), seq
            
    
def mut_df_chk_exactmatch(
    df: pd.DataFrame, 
    motif_chklist: list = [],
    chrom_key: str = 'Chromosome',
    loc_key: str = 'hg38_start',
    seqID_key: str = 'Tumor_Seq_Allele2',
):
    res = {k: [] for k in motif_chklist}
    for i, r in df.iterrows():
        for ii, m in enumerate(motif_chklist):
            if ii==0:
                out, seq = match_mutVquery(
                    alt_seq=r[seqID_key], 
                    chrom=r[chrom_key], 
                    loc=r[loc_key],
                    motif=m,
                    fasta_seq=fasta_seq,)
                res[m].append(out)
            else:
                res[m].append(seq_match(seq, m))
    for k, v in res.items():
        df[k + '_match_' + seqID_key] = v
    return df
        
def enrichment_tst(
    dfAB: pd.DataFrame,
    dfCD: pd.DataFrame,
    condAB: tuple = ('alt_seq_apORfox', 1, 'alt_seq_apORfox', 0),
    condCD: tuple or None = ('y_pred', 1, 'y_pred', 0),
    verbose: bool = True,
) -> "p, table" :
    """Enrichment test (chi2_contingency) and OR = A/B / C/D for tab = [[A, B], [C, D]]
    
    """
    from scipy.stats import chi2_contingency
    if condCD is None:
        condCD = condAB
    table = np.array(
        [
         [sum(dfAB[condAB[0]]==condAB[1]), sum(dfAB[condAB[2]]==condAB[3])],
         [sum(dfCD[condCD[0]]==condCD[1]), sum(dfCD[condCD[2]]==condCD[3])],
        ])
    if verbose:
        print('\nContingency table:\n', ' ', table)
    OR = (table[0, 0] / table[0, 1]) / (table[1, 0] / table[1, 1])
    if verbose:
        print(f"OR: {OR:.4f}")
    chi2, p, dof, expected = chi2_contingency(table)
    if verbose:
        print(f"P-val: {p:.4e}")
    return p, table


if __name__ == '__main__':
    df = get_mut_df('brca')
    dt = mut_df_chk_exactmatch(df, ['TGAGTCA', ])
    enrichment_tst(
        dfAB=dt.loc[dt['distance_from_summit'] <= 250, :],
        dfCD=dt.loc[dt['distance_from_summit'] > 250, :],
        condAB=('TGAGTCA_match_Tumor_Seq_Allele2', True, 'TGAGTCA_match_Tumor_Seq_Allele2', False),
        condCD=None,
    )