import pandas as pd
from argparse import ArgumentParser
import pybedtools
from pybedtools import BedTool
import re


def ld_expand (var,  ld_anno, ld):    
    try:
        var_id = ld_anno.loc[(ld_anno['Position'] == var['hg38_end']) & (ld_anno['ALT'] == var['alt']),'Uniq_ID'].item()
    except:
        return None
    ld_snps = set(ld.loc[(ld['Uniq_ID_1'] == var_id) ,'Uniq_ID_2'].values)
    ld_snps2 = set(ld.loc[(ld['Uniq_ID_2'] == var_id) ,'Uniq_ID_1'].values)
    ld_snps = ld_snps.union(ld_snps2)
    ld_snps.add(var_id)

    out_df = ld_anno.loc[ld_anno['Uniq_ID'].isin(ld_snps),['Position','REF','ALT','Uniq_ID','rsID']]
    out_df['chrom'] = var['chrom']
    out_df['start'] = out_df['Position'] - 1
    out_df['lead_snp'] = var_id
    out_df = out_df.loc[:,['chrom','start', 'Position','REF','ALT','Uniq_ID','rsID','lead_snp']]
    out_df.rename(columns={'Position':'end'}, inplace=True)    
    return(out_df)


parser = ArgumentParser()
parser.add_argument("--snp_file")
parser.add_argument("--peak_file")
parser.add_argument("--chrom")
parser.add_argument("--dir", default="/illumina/scratch/deep_learning/asalcedo/scATAC/sum_stats_ld_expanded_bed/")

args = parser.parse_args()
in_chrom= args.chrom
peaks_bedfile= args.peak_file
outdir = args.dir
snpfile_base = re.sub("/.*/","", args.snp_file)


peakfile_base = re.sub("/.*/","", peaks_bedfile)
peakset = re.sub("(^[A-Z]*?)_","", peakfile_base)
peakset = peakset.replace(".bed","")

#LD expansion is based on topmed LD tables
topmed_dir='/illumina/scratch/deep_learning/asalcedo/scATAC/topmed_ld/'
ss_df = pd.read_csv(args.snp_file, sep="\t")


ld_anno = pd.read_csv(topmed_dir + 'EUR_' + in_chrom + '_no_filter_0.2_1000000_info_annotation.csv.gz')
ld = pd.read_csv(topmed_dir + 'EUR_' + in_chrom + '_no_filter_0.2_1000000_LD.csv.gz')

expanded_snp_list = []
for i,x in ss_df.loc[ss_df['chrom'] == in_chrom,:].iterrows():
    tmp = ld_expand(x, ld_anno, ld)
    expanded_snp_list.append(tmp)
ss_df_expanded = pd.concat(expanded_snp_list)

ss_expanded_file = snpfile_base.replace(".bed", "_expanded.bed")
ss_df_expanded.to_csv(outdir +  ss_expanded_file, index=False, header=False, sep="\t")
ss_expanded_bed = pybedtools.BedTool(outdir +  ss_expanded_file)
peaks_bed  = pybedtools.BedTool(peaks_bedfile)

peak_snps = peaks_bed.intersect(ss_expanded_bed,  wa=True, wb=True)
peak_snps_df = peak_snps.to_dataframe()    
peak_snps_df = peak_snps_df.iloc[:,[3,4,5,6,7,8,9,10,0,1,2]]
peak_snps_df.columns = ['snp_chrom', 'snp_start','snp_end','ref', 'alt', 'id', 'rsid', 'lead_snp','peak_chrom', 'peak_start', 'peak_end']
outfile = outdir + snpfile_base.replace(".bed", "") + "_" + in_chrom + "_expanded_" + peakset + "_peak_overlap.bed"
outfile_unique = outdir + snpfile_base.replace(".bed", "") + "_" + in_chrom + "_expanded_" + peakset + "_peak_overlap_unique.bed"
peak_snps_df.to_csv(outfile, sep="\t", index=False)

peak_snps_df.drop('lead_snp', axis=1, inplace=True)
peak_snps_df.drop_duplicates().to_csv(outfile_unique, sep="\t", index=False)

