from pyfaidx import Fasta
import pandas as pd

sizesfile = "/illumina/scratch/deep_learning/akumar22/toymodel/master_files/hg38.chrom.sizes"
#gnomad_file = "/illumina/scratch/deep_learning/akumar22/generated_data/gnomad.genomes.r3.0.sites.af.txt"
gnomad_file = "/illumina/scratch/deep_learning/public_data/gnomad/r3.0/vcf/genomes/gnomad.genomes.r3.0.sites.af.txt.gz"
fasta_file = '/illumina/scratch/deep_learning/lsundaram/singlecelldatasets/TCGA/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta'
fasta_seq=Fasta('/illumina/scratch/deep_learning/lsundaram/singlecelldatasets/TCGA/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta')
refcomplement = {"A":"T",
                    "T":"A",
                    "G":"C",
                    "C":"G"}
rmsk_file = "/illumina/scratch/deep_learning/akumar22/TCGA/mutations_scoring/master_files/rmsk.bed"
phylop_file = "/illumina/scratch/deep_learning/public_data/refdata/hg38/multiple_alignments/phyloP100way/hg38.phyloP100way.bw"

def get_promoterai_tss():
    tss_df = pd.read_csv("/illumina/scratch/deep_learning/nersaro/promoterAI/data/ref_data/gencodev39_cage_ratio_to_sum_refined_tss_positions_transcripts_protein_coding_inclZeros_withTranscriptID.tsv",sep="\t")
    tss_df = tss_df[["chrom","tss_pos","gene"]]
    tss_df["end"] = tss_df["tss_pos"] + 1
    tss_df = tss_df[["chrom","tss_pos","end","gene"]]
    tss_df.columns = ["chr","start","end","gene"]
    return tss_df

def get_gencode_tss(protein_coding_only=False):
    tss_file = "/illumina/scratch/deep_learning/akumar22/TCGA/mutations_scoring/master_files/gencode.v37.tss.tsv"
    tss_df = pd.read_csv(tss_file,sep="\t")
    
    if protein_coding_only:
        tss_df = tss_df[tss_df["transcript_type"].isin(["protein_coding"])].reset_index(drop=True)

    tss_df["end"] = tss_df["tss_pos"] + 1
    tss_df = tss_df[["chrom","tss_pos","end","gene"]].sort_values(["chrom","tss_pos"]).reset_index(drop=True)
    tss_df = tss_df[tss_df["gene"].apply(pd.isna) == 0]
    
    tss_df.columns = ["chr","start","end","gene"]
    return tss_df

def get_cosmic_pancan_genes():

    cosmos_pancan_file = "/illumina/scratch/deep_learning/asalcedo/cancer_gene_census.csv"
    cosmos_pancan_df = pd.read_csv(cosmos_pancan_file)
    pancan_genes = cosmos_pancan_df[~cosmos_pancan_df["Role in Cancer"].isin(["fusion"])]["Gene Symbol"].tolist()
    return pancan_genes

def get_ogtsg():
    cosmos_pancan_file = "/illumina/scratch/deep_learning/asalcedo/cancer_gene_census.csv"
    cosmos_pancan_df = pd.read_csv(cosmos_pancan_file)

    cosmos_pancan_df = cosmos_pancan_df[(~cosmos_pancan_df["Role in Cancer"].isna()) & 
                                        (cosmos_pancan_df["Role in Cancer"]!="fusion")]

    pancan_genes = list(cosmos_pancan_df["Gene Symbol"].unique())
    return pancan_genes
