from tqdm import tqdm
import pickle

def get_hg38_splits():
    hg38_splits = {}
    hg38_splits[0]={'test':['chr1'],
                    'valid':['chr10','chr8'],
                    'train':['chr2','chr3','chr4','chr5','chr6','chr7','chr9','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX','chrY']}

    hg38_splits[1]={'test':['chr19','chr2'],
                    'valid':['chr1'],
                    'train':['chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr20','chr21','chr22','chrX','chrY']}

    hg38_splits[2]={'test':['chr3','chr20'],
                    'valid':['chr19','chr2'],
                    'train':['chr1','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr21','chr22','chrX','chrY']}

    hg38_splits[3]={'test':['chr13','chr6','chr22'],
                    'valid':['chr3','chr20'],
                    'train':['chr1','chr2','chr4','chr5','chr7','chr8','chr9','chr10','chr11','chr12','chr14','chr15','chr16','chr17','chr18','chr19','chr21','chrX','chrY']}

    hg38_splits[4]={'test':['chr5','chr16','chrY'],
                    'valid':['chr13','chr6','chr22'],
                    'train':['chr1','chr2','chr3','chr4','chr7','chr8','chr9','chr10','chr11','chr12','chr14','chr15','chr17','chr18','chr19','chr20','chr21','chrX']}
    return hg38_splits

hg38_splits = get_hg38_splits()

def create_splits(all_seq,train_split,val_split,test_split,chm_colname):

    assert type(train_split) == list
    assert type(val_split) == list
    assert type(test_split) == list

    X_train=all_seq[all_seq[chm_colname].isin(train_split)]
    X_val=all_seq[all_seq[chm_colname].isin(val_split)]
    X_test=all_seq[all_seq[chm_colname].isin(test_split)]

    return X_train, X_val, X_test



count_gc = lambda seq: (seq.count("G") + seq.count("C"))/len(seq)
def compute_gc_bed(input_df,fasta_seq,input_width,roundoff=None):
    gcs=[]
    for idx,i in tqdm(enumerate(input_df.values), total=len(input_df)):
        chm = str(i[0])
        start = int(i[1])
        end = int(i[2])
        summit = (start + end)//2
        ns = summit - input_width // 2
        ne = ns + input_width
        gc_local=count_gc(str(fasta_seq[chm][ns:ne]))
        if roundoff == None:
            gcs.append(gc_local)
        else:
            gcs.append(round(gc_local,roundoff))
    return gcs


def write_pickle(df,filename):
    with open(filename, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def read_pickle(filename):
    df = None
    with open(filename, 'rb') as handle:
        df = pickle.load(handle)
    return df
