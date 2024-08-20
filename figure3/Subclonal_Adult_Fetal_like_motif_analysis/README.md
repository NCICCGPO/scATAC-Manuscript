This folder contains code for doing the motif enrichment for differential peaks in GBM subclones and understanding the enrichment of the TF binding site.


The gbm_subclones_adult_fetal_motifs.ipynb file contains the enrichment test for the neural network-clean Vierstra motif sets between the differential peaks between the fetal and adult subclones in GBM45 and GBM39 samples. The reported enrichment odds ratios are averaged across the fetal and adult subclones in Fig 3.


The gbm_subclone_tf_enrichment_chr6.ipynb file contains the code to compute the enrichment of TF binding sites for chromosome 6 TFs in the GBM45 sample near other TF genes in other chromosomes in copy-neutral regions compared to other genes. The definition of TF vs. non-TF gene characterization is obtained from https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv. 
