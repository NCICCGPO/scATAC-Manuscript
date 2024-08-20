if __name__ == '__main__':
    import argparse
    import pickle
    import pandas as pd
    import glob
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help="point to results from genloc_usm.py files")
    parser.add_argument('--export', default=None, help="specify to save file")
    args = parser.parse_args()

    import mutationism_score as glism

    res_files = glob.glob(os.path.join(args.output_dir, '{}_chunk*.pkl'.format('*')))
    eval_ = glism.eval_glism(res_files=res_files).agg_res_files()
    id_cols = ['Chromosome', 'hg38_start', 'hg38_end', 'mutation_id', 'sample']
    eval_ = pd.pivot(eval_, index=id_cols, columns='seq_type', values='log2FC_seqVshuffled').reset_index()
    eval_['ref-mut'] = eval_['ref'] - eval_['mut']
    print(eval_.head())

    if args.export is not None:
        # with open(export, 'wb') as f:
        #     pickle.dump(eval_, f, protocol=pickle.HIGHEST_PROTOCOL)
        eval_.to_csv(args.export)
        print('Aggregated df written to:', eval_)