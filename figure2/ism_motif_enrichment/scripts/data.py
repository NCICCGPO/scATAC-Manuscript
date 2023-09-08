"""
Organization:
  

References:

  di-nt shuffle:
    code: https://github.com/wassermanlab/BiasAway/blob/master/altschulEriksonDinuclShuffle.py
      by P. Clote, Oct 2003
"""
import pandas as pd
import os
import pickle
import glob
import random
import sys
import numpy as np

def can_job_be_submitted(
    user: str = 'nravindra',
    max_n_jobs_per_user: int = 3000,
    safety_cushion: int = 50,
):
    """is job under number of job limits. 
    - TODO: add other conditions
    
    Arguments:
      
    """
    # os.system(f"qstat | grep nravindra | wc -l")
    command = "qstat -u {} | grep {} | wc -l".format(user, user)
    n_jobs = int(os.popen(command).read().split()[0])
    if n_jobs < max_n_jobs_per_user - safety_cushion:
        return True
    else:
        return False

def computeCountAndLists(s):
  #WARNING: Use of function count(s,'UU') returns 1 on word UUU
  #since it apparently counts only nonoverlapping words UU
  #For this reason, we work with the indices.

  #Initialize lists and mono- and dinucleotide dictionaries
  List = {} #List is a dictionary of lists
  List['A'] = []; List['C'] = [];
  List['G'] = []; List['T'] = [];
  nuclList   = ["A","C","G","T"]
  s       = s.upper()
  s       = s.replace("T","T")
  nuclCnt    = {}  #empty dictionary
  dinuclCnt  = {}  #empty dictionary
  for x in nuclList:
    nuclCnt[x]=0
    dinuclCnt[x]={}
    for y in nuclList:
      dinuclCnt[x][y]=0

  #Compute count and lists
  nuclCnt[s[0]] = 1
  nuclTotal     = 1
  dinuclTotal   = 0
  for i in range(len(s)-1):
    x = s[i]; y = s[i+1]
    List[x].append( y )
    nuclCnt[y] += 1; nuclTotal  += 1
    dinuclCnt[x][y] += 1; dinuclTotal += 1
  assert (nuclTotal==len(s))
  assert (dinuclTotal==len(s)-1)
  return nuclCnt,dinuclCnt,List
 
 
def chooseEdge(x,dinuclCnt):
  numInList = 0
  for y in ['A','C','G','T']:
    numInList += dinuclCnt[x][y]
  z = random.random()
  denom=dinuclCnt[x]['A']+dinuclCnt[x]['C']+dinuclCnt[x]['G']+dinuclCnt[x]['T']
  numerator = dinuclCnt[x]['A']
  if z < float(numerator)/float(denom):
    dinuclCnt[x]['A'] -= 1
    return 'A'
  numerator += dinuclCnt[x]['C']
  if z < float(numerator)/float(denom):
    dinuclCnt[x]['C'] -= 1
    return 'C'
  numerator += dinuclCnt[x]['G']
  if z < float(numerator)/float(denom):
    dinuclCnt[x]['G'] -= 1
    return 'G'
  dinuclCnt[x]['T'] -= 1
  return 'T'


def connectedToLast(edgeList,nuclList,lastCh):
  D = {}
  for x in nuclList: D[x]=0
  for edge in edgeList:
    a = edge[0]; b = edge[1]
    if b==lastCh: D[a]=1
  for i in range(2):
    for edge in edgeList:
      a = edge[0]; b = edge[1]
      if D[b]==1: D[a]=1
  ok = 0
  for x in nuclList:
    if x!=lastCh and D[x]==0: return 0
  return 1
 

def eulerian(s):
  nuclCnt,dinuclCnt,List = computeCountAndLists(s)
  #compute nucleotides appearing in s
  nuclList = []
  for x in ["A","C","G","T"]:
    if x in s: nuclList.append(x)
  #compute numInList[x] = number of dinucleotides beginning with x
  numInList = {}
  for x in nuclList:
    numInList[x]=0
    for y in nuclList:
      numInList[x] += dinuclCnt[x][y]
  #create dinucleotide shuffle L 
  firstCh = s[0]  #start with first letter of s
  lastCh  = s[-1]
  edgeList = []
  for x in nuclList:
    if x!= lastCh: edgeList.append( [x,chooseEdge(x,dinuclCnt)] )
  ok = connectedToLast(edgeList,nuclList,lastCh)
  return ok,edgeList,nuclList,lastCh


def shuffleEdgeList(L):
  n = len(L); barrier = n
  for i in range(n-1):
    z = int(random.random() * barrier)
    tmp = L[z]
    L[z]= L[barrier-1]
    L[barrier-1] = tmp
    barrier -= 1
  return L


def dinuclShuffle(s):
  ok = 0
  while not ok:
    ok,edgeList,nuclList,lastCh = eulerian(s)
  nuclCnt,dinuclCnt,List = computeCountAndLists(s)

  #remove last edges from each vertex list, shuffle, then add back
  #the removed edges at end of vertex lists.
  for [x,y] in edgeList: List[x].remove(y)
  for x in nuclList: shuffleEdgeList(List[x])
  for [x,y] in edgeList: List[x].append(y)

  #construct the eulerian path
  L = [s[0]]; prevCh = s[0]
  for i in range(len(s)-2):
    ch = List[prevCh][0] 
    L.append( ch )
    del List[prevCh][0]
    prevCh = ch
  L.append(s[-1])
  return ''.join(L)
  # t = string.join(L,"")
  # return t


def get_fasta(fasta_file: str = './tmp//singlecelldatasets/TCGA/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta'):
    """Point to fasta file, get a FAsta InDeX object to query.
    
    Description:
      See https://github.com/mdshw5/pyfaidx codebase for documentation. Here, we 
      pyfaidx to take advantage of samtools faidx function to handle sequences 
      efficiently.
      
    Arguments:
      fasta_file: `str` (optional, default: '*/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta')
        Reference sequence
    
    Returns:
      fasta_seq: dict
        a pyfaidx class object that functions like a dictionary
    """
    
    from pyfaidx import Fasta
    return Fasta(fasta_file) 


def load_peak_locs(csv_file: str = './tmp//singlecelldatasets/TCGA/motifs/BRCA/vierstra_Archetype_BRCAagg.csv',
                   chunksize: int = 1e6,
                   verbose: bool = False):
    """Load a csv file of motif name, seqnames, start, and end in chunks
    
    Arguments:
      csv_file: str (optional, default: ~/BRCA/vierstra_Archetype_BRCAagg.csv')
        a motif file with columns specifying some group_name, seqnames, start, and end locs
      chunksize: int (optional, default: 1e6)
        set how many rows to be read into RAM
      verbose: bool (optional, default: False)
        if true, show what the colnames are to make sure 
    """
    df_reader = pd.read_csv(csv_file, chunksize=chunksize)
    return df_reader

def one_hot_encode(seqs):
    """Get one hot encoding in numpy floats for sequences
    
    Arguments:
      seqs: list or str
        Can be list of str sequences or seqs as str like 'ATGC', treated
          equivalently to 'atgc' or ['ATGC']. NOTE: all seqs must be of same 
          length.
    
    Returns: 
      out: np.ndarray
    """
    def get_base_encoding():
        base_encoding = np.zeros((ord('z') + 1, 1), dtype=int)
        base_encoding[ord('A')] = 1
        base_encoding[ord('C')] = 2
        base_encoding[ord('G')] = 3
        base_encoding[ord('T')] = 4
        base_encoding[ord('a')] = 1
        base_encoding[ord('c')] = 2
        base_encoding[ord('g')] = 3
        base_encoding[ord('t')] = 4
        return base_encoding
    IN_MAP = np.asarray(
        [[0,0,0,0],
         [1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1]], 
        dtype='float')
    base_encoding = get_base_encoding()
    if isinstance(seqs, str):
        seqs = [seqs]
    int_seqs = np.concatenate([base_encoding[np.frombuffer(str(s).encode(),dtype=np.byte,count=len(s))] for s in seqs],axis=1).T
    out = IN_MAP[int_seqs]
    return out

def get_refseq_window(fasta_seq, 
                      chrom: int, start: int, end: int,
                      output_length: int = 1364,
                      jitter: int = 0,
                      motif_key: str = 'group_name', 
                      chrom_key: str = 'seqnames', 
                      start_key: str = 'start', 
                      end_key: str = 'end',
                      return_peak_midpt_loc: bool = False
                     ):
    """Get the reference sequence from a pyfaidx object of equal surrounding width from peak middle
         unless jitter is non-zero.
    
    Arguments:
      fast_seq: pyfaidx class 
        output of get_fasta, a queryable sequence class
      output_length: int (optional, default: 1364)
        specify the window size
      chrom: str
        all lowercase chromosome key to query the fasta_seq
      start: int
        coord of start of motif or peak
      end: int
        coord of end of motif or peak
    """
    peak_midpt = (end - start) // 2 + start 
    peak_midpt += np.random.choice([0, 1]) if (end - start) % 2 == 1 else 0
    peak_midpt += np.random.choice([0, 1]) if peak_midpt % 2 == 0 and output_length % 2 == 0 else 0
    pad = 1 if output_length % 2 == 1 else 0
    left = int(peak_midpt - (output_length // 2))
    right = int(peak_midpt + (output_length // 2) + pad)

    seq = str(fasta_seq[chrom][left:right])
    
    assert len(seq) == output_length, 'left:right indexing is not correct'
    # assert seq still in same chromosome
    if return_peak_midpt_loc:
      return seq, peak_midpt
    else:
      return seq






if __name__ == '__main__':
    
    # motif locations
    brca_motifs = './tmp//singlecelldatasets/TCGA/motifs/BRCA/vierstra_Archetype_BRCAagg.csv'
    
