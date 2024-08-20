import pandas as pd
import pysam
import glob
import numpy as np
import argparse

def get_vaf(rec, bamfile):
    region = rec['chrom'] + ":" + str(rec['pos']) + "-" + str(rec['pos'])
    ref = rec['ref']
    alt = rec['alt']
    count_dic = {'A':0, 'C':0,'G':0,'T':0}
    if len(pysam.mpileup("-r", region, bamfile)) == 0:
        return region, rec['ref'], rec['alt'], np.nan, np.nan
    for x in  pysam.mpileup("-r", region, bamfile).split('\t')[4].upper():
        if x in [ref, alt]:
            count_dic[x] +=1
    if count_dic[ref] < 1:
        return region, rec['ref'], rec['alt'], np.nan, np.nan
    vaf = count_dic[alt]/(count_dic[alt] + count_dic[ref])
    total_dp = (count_dic[alt] + count_dic[ref])
    return region, rec['ref'], rec['alt'], vaf, total_dp


parser = argparse.ArgumentParser()
parser.add_argument('--case_id')
parser.add_argument('--use_wgs', action='store_true')
args = parser.parse_args()

case_id = args.case_id
print(case_id)

all_files = glob.glob("./snv_cancer_peak_overlap_all_samples/TCGA-*overlap.txt", recursive=True) ### TODO: point to correct path with CaVEMan
snv_file_df = pd.DataFrame({'region_file':all_files})
snv_file_df['case_id'] = snv_file_df['region_file'].str.extract(".*/(TCGA-.*?)\.")

common_snv = None
for snv_file in snv_file_df.loc[snv_file_df['case_id'] == case_id,'region_file'].values:
    snv_df = pd.read_csv(snv_file, sep="\t")
    snv_df['var_id'] = snv_df['chrom'] + "_" + snv_df['pos'].astype('str')
    if common_snv is None:
        common_snv = set(snv_df['var_id'].values)
    else:
        common_snv = common_snv.intersection(set(snv_df['var_id'].values))

snv_df = snv_df[snv_df['var_id'].isin(common_snv)]

atac_ss = pd.read_csv('190101_Samples_For_scATAC.csv', sep=",") ## TODO: point to correct path
atac_ss['uuid'] = atac_ss['UUID_SampleName Prefix'].str.replace("-","_")

scATAC_files = glob.glob("path to scatac bams/*.bam", recursive=False) ### TODO: path to scATAC bams
bam_file_df = pd.DataFrame({'bam_file':scATAC_files})
bam_file_df['UUID'] = bam_file_df['bam_file'].str.extract(".*/scATAC_(.*?)_X0")
bam_file_df2 = pd.merge(bam_file_df, atac_ss, left_on="UUID", right_on="uuid")
atac_bamfile = bam_file_df2.loc[bam_file_df2['submitter_id'] == case_id,'bam_file'].values[0]

atac_vafs = snv_df.apply(lambda x: get_vaf(x, atac_bamfile), axis=1, result_type='expand')
atac_vafs.columns = ['region','ref','alt', 'atac_vaf','atac_dp']

if args.use_wgs:
    wgs_bamfiles = glob.glob("path to bams /*.bam", recursive=True) ## TODO: paths to wgs bams
    wgs_bam_df = pd.DataFrame({'file':wgs_bamfiles})
    wgs_bam_df['case_id'] = wgs_bam_df['file'].str.extract("/(TCGA.*?).bam")
    wgs_bamfile = wgs_bam_df.loc[wgs_bam_df['case_id']== case_id,'file'].values[0]
    wgs_vafs= snv_df.apply(lambda x: get_vaf(x, wgs_bamfile), axis=1, result_type='expand')
    wgs_vafs.columns = ['region','ref','alt', 'wgs_vaf','wgs_dp']
    all_vafs = pd.merge(wgs_vafs, atac_vafs,  left_on=['region','ref','alt'], right_on=['region','ref','alt'])
else:
    all_vafs = atac_vafs


all_vafs.to_csv(case_id + "_scATAC_vafs.txt", sep="\t", index=False)
