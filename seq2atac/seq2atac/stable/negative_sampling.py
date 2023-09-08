import pandas as pd
from tqdm import tqdm

def create_matched_master(peaks_bed, nopeaks_bed, model_dimension, save_path_csv):
    ## peak_csv_df, nopeaks_df are in bed format
    ## match on columns [0,3:]
    ##

    peak_csv_df = pd.read_csv(peaks_bed, sep="\t", header=None)
    nopeaks_df = pd.read_csv(nopeaks_bed, sep="\t", header=None)
    assert peak_csv_df.shape[1] == nopeaks_df.shape[1], "peak and negatives have differing columns"
    all_columns = list(range(peak_csv_df.shape[1]))

    groupby_columns = [col for col in all_columns if col not in [1,2]]
    print("Grouping by: ",groupby_columns)
    peak_csv_df_group = peak_csv_df.groupby(groupby_columns).size().reset_index(name="counts").sort_values(groupby_columns)

    negatives_df = []
    positives_df = []
    
    all_chms = [f"chr{num}" for num in range(1,23)] + ["chrX"]

    for chm in all_chms:
        print(chm)
        peak_csv_df_group_chm = peak_csv_df_group[peak_csv_df_group[0]== chm]
        nopeak_sorted_chm = nopeaks_df[nopeaks_df[0]==chm]
        peak_chm = peak_csv_df[peak_csv_df[0]==chm]

        for idx,row in tqdm(peak_csv_df_group_chm.iterrows(), total=len(peak_csv_df_group_chm)):
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

            df_to_add = total_matches.sample(n=num_samples)
            pos_df_to_add = pos_total_matches.sample(n=num_samples)

            negatives_df.append(df_to_add)
            positives_df.append(pos_df_to_add)

    positives_df = pd.concat(positives_df).reset_index(drop=True)
    positives_df["Summit"] = ((positives_df[1] + positives_df[2])/2).astype(int)
    positives_df[1] = positives_df["Summit"] - model_dimension//2
    positives_df[2] = positives_df[1] + model_dimension

    negatives_df = pd.concat(negatives_df).reset_index(drop=True)
    negatives_df["Summit"] = ((negatives_df[1] + negatives_df[2])/2).astype(int)
    negatives_df[1] = negatives_df["Summit"] - model_dimension//2
    negatives_df[2] = negatives_df[1] + model_dimension

    master_df = pd.concat([positives_df[[0,1,2]], negatives_df[[0,1,2]]], axis=1, ignore_index=True)
    master_df.columns = ["peak_chr","peak_start","peak_end","neg_chr","neg_start","neg_end"]
    master_df["peak_label"] = 1
    master_df["neg_label"] = 0

    master_df.to_csv(save_path_csv,index=None)
    return master_df
