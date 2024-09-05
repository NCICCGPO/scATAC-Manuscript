allelic_imbalance folder contains the get vaf code to compute the read coverage for the reference and alternate snps from the scATAC bams and WGS bams.

ldsc folder contains the code to run the LDSC regression analysis of BCAC, UK biobank and Finngen BRCA cohorts presented in figure 5. The ldsc was performed based on https://github.com/bulik/ldsc. The datasets were obtained from the respective consoritum data portals.

pcawg_variant_prioritization contains the code for running the non coding somatic mutation enrichment analysis in PCAWG. The TSS file for the nearest gene annotation "gencodev39_cage_ratio_to_sum_refined_tss_positions_transcripts_protein_coding_inclZeros_withTranscriptID.tsv" is available in the publication page. The definitions of oncogene and tumor suppressor genes were obtained from Cosmic. 

vignette folder contains the plotting tools for the per base importance scores and creating the vignettes used in the paper.
