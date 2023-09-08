import numpy as np
import pandas as pd
from tqdm import tqdm

def create_matched_master(peak_csv_df, nopeaks_df, groupby_columns, return_unmatched=False, verbose=True):
    ### make sure that peak_csv_df, nopeaks_df are non-overlapping in groupby_columns
    ## peak_csv_df, nopeaks_df are in bed format
    ## match on columns [0,3:]
    ##
    np.random.seed(94404)
    print("Grouping by: ",groupby_columns)
    peak_csv_df_group = peak_csv_df.groupby(groupby_columns).size().reset_index(name="counts").sort_values(groupby_columns)
    negatives_df = []
    positives_df = []
    
    main_col = groupby_columns[0]
    all_chms = list(set(peak_csv_df[main_col].tolist()))

    for chm in all_chms:
        peak_csv_df_group_chm = peak_csv_df_group[peak_csv_df_group[main_col]== chm]
        nopeak_sorted_chm = nopeaks_df[nopeaks_df[main_col]==chm]
        peak_chm = peak_csv_df[peak_csv_df[main_col]==chm]

        for idx,row in tqdm(peak_csv_df_group_chm.iterrows(), total=len(peak_csv_df_group_chm), disable= not verbose):
        # for idx,row in peak_csv_df_group_chm.iterrows():
            # chm_ = row[0]
            # count = row["counts"]
            to_match = row[1:-1]

            total_matches = nopeak_sorted_chm
            for k,v in to_match.items():
                total_matches = total_matches[total_matches[k] == v]

            pos_total_matches = peak_chm
            for k,v in to_match.items():
                pos_total_matches = pos_total_matches[pos_total_matches[k] == v]

            num_samples = min(len(pos_total_matches),len(total_matches))

            df_to_add = total_matches.sample(n=num_samples,random_state=94404)
            pos_df_to_add = pos_total_matches.sample(n=num_samples,random_state=94404)

            negatives_df.append(df_to_add)
            positives_df.append(pos_df_to_add)

    positives_df = pd.concat(positives_df)
    negatives_df = pd.concat(negatives_df)
    
    if return_unmatched:
        unmatched_pos = peak_csv_df[~peak_csv_df.index.isin(positives_df.index)]
        unmatched_neg = nopeaks_df[~nopeaks_df.index.isin(negatives_df.index)]
        return positives_df.reset_index(drop=True), negatives_df.reset_index(drop=True),\
                unmatched_pos.reset_index(drop=True), unmatched_neg.reset_index(drop=True)
    
    return positives_df.reset_index(drop=True),negatives_df.reset_index(drop=True)


def matching_logic(somatic_df, control_df, levels, verbose=True):   

    assert type(levels) == list
    assert sum([type(level) == list for level in levels]) == len(levels)

    somatic_matched_list = []
    control_matched_list = []

    somatic_unmatched = somatic_df.copy()
    control_unmatched = control_df.copy()

    for level in levels:
        somatic_matched, control_matched, somatic_unmatched, control_unmatched = create_matched_master(somatic_unmatched, control_unmatched, level, return_unmatched=True, verbose=verbose)
        somatic_matched_list.append(somatic_matched)
        control_matched_list.append(control_matched)
        if not len(somatic_unmatched):
            break

        if not len(control_unmatched):
            break

    somatic_matched = pd.concat(somatic_matched_list,axis=0,ignore_index=True)
    control_matched = pd.concat(control_matched_list,axis=0,ignore_index=True)
    
    return somatic_matched, control_matched
