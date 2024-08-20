import glob
import os
import pandas as pd
import numpy as np
import pickle
import time
import re
import itertools


class eval_ism():
    def __init__(
        self,
        filepath: list or str,
        outfile: str or None = None,
        name_key: str = 'group_name',
        chrom_key: str = 'seqnames',
        start_key: str = 'start',
        end_key: str = 'end',
        method: str = 'ave_over_folds', 
        clean_mode: bool = False,
        debug_mode: bool = False,
    ):
        self.filepath = filepath
        self.outfile = outfile
        self.name_key = name_key
        self.chrom_key = chrom_key
        self.start_key = start_key
        self.end_key = end_key
        self.method = method
        self.clean_mode = clean_mode
        self.debug_mode = debug_mode


    def eval_chunk_df(
        self,
        chunk_df: pd.DataFrame, 
        ):

        if self.method == 'ave_over_folds':
            chunk_df['model_ave'] = chunk_df.loc[:, [i for i in chunk_df.columns if len(re.findall('(model)(.*)(out)', i)) > 0]].mean(1)
            
            # main effect
            chunk_reduced = chunk_df.groupby([self.name_key, self.chrom_key, self.start_key, self.end_key, 'source']).mean().reset_index()
            # # recalc aves to first reduce over shuffled seqs
            # chunk_reduced['model_ave'] = chunk_reduced.loc[:, [i for i in chunk_reduced.columns if len(re.findall('(model)(.*)(out)', i)) > 0]].mean(1)
            
            ## calc diffsn 
            chunk_diff = chunk_reduced.loc[
                chunk_reduced['source']=='seq', 
                [self.name_key, self.chrom_key, self.start_key, self.end_key, 'model_ave']].merge(
                chunk_reduced.loc[
                    chunk_reduced['source']=='shuffled_seqs', 
                    [self.name_key, self.chrom_key, self.start_key, self.end_key, 'model_ave']],
                on=[self.name_key, self.chrom_key, self.start_key, self.end_key,], 
                suffixes=['_seq', '_shuffled'])
            
            # null distribution
            if True:
                null_dist = []
                dt = chunk_df.loc[chunk_df['source']=='shuffled_seqs'].groupby([self.name_key, self.chrom_key, self.start_key, self.end_key])
                # n = dt.size().shape[0]
                for i, (idx, grp) in enumerate(dt):
                    a = grp['model_ave'].to_numpy()
                    null_dist.append(np.mean([np.log2(a[i]) - np.log2(a[j]) for i, j in list(itertools.combinations(list(range(a.shape[0])), 2))]))
        else:
            raise NotImplementedError 

        return chunk_diff, null_dist
    
    def proc_motifism_file(
        self,
        ):

        if self.debug_mode:
            print(self.outfile)
        
        chunk_res_agg = pd.DataFrame()
        null_dists = []

        print('Checking for chunks at:', os.path.join(self.filepath, 'chunk*.pkl'))

        chunk_list = glob.glob(os.path.join(self.filepath, 'chunk*.pkl'))
        print('found {} files. Starting processing...'.format(len(chunk_list)))
        
        for i, f in enumerate(chunk_list):
            chunk_name = os.path.split(f)[1].split('.pkl')[0]
            chunk = pd.read_pickle(f)
            df, null = self.eval_chunk_df(chunk)
            chunk_res_agg = pd.concat([chunk_res_agg, df]) # ignore_index = True?
            null_dists += null
            
            if self.clean_mode and False: # failsafe!!! Don't delete these files until someone complains about mem
                os.remove(f)

            print('...  done with:', f)
                
        if self.outfile is not None:
            with open(self.outfile, 'wb') as f:
                pickle.dump({'df': chunk_res_agg, 'null_dists': null_dists}, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
        return chunk_res_agg, null_dists

def main(
    filepath: str,
    outfile: str or None = None,
    ):

    if outfile is not None and not os.path.exists(os.path.split(outfile)[0]):
        os.makedirs(os.path.split(outfile)[0])

    evaluator = eval_ism(filepath=filepath, outfile=outfile)
    return evaluator.proc_motifism_file()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--filepath', type=str,
                        help="filepath to chunked .pkl result files to process. Can specify sample shorthand too, e.g., brca10")
    args = parser.parse_args()

    print('\nArguments passed:')
    print(args)

    valid_file_specs = [
        'blca', 'skcm', 'gbm', 
        'kirp', 'kirc', 'luad', 'coad',

        'GBM39_cloneA', 'GBM39_cloneB', 
        'GBM45_cloneA', 'GBM45_cloneB',
        ]
    
    valid_file_specs = [s.lower() for s in valid_file_specs]
    
    if args.filepath.lower() in valid_file_specs or 'brca' in args.filepath.lower():
        g = args.filepath.lower() # note: motifism converts all to lowercase
        args.filepath = './ism_cleaning_output/{}/'.format(g) ### TODO: path from motifism.py script
        assert os.path.exists(args.filepath), 'point to valid filepath'
        outfile = './results/{}.pkl'.format(g)

        df, null = main(
            filepath=args.filepath,
            outfile=outfile,
            )

        print('Done with {}. Outfile:'.format(g), outfile)
