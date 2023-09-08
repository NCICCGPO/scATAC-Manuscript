TCGA
Contains scripts and notebooks for analyses presented in Single-cell analysis of open chromatin landscapes across diverse cancer types.

Requirements Apart from requirements listed in seq2atac/requirements.txt, please install the following to ensure consistency

kerasAC:
shap:
seq2atac contains core helpers and model training functions that are used by other analyses. Install using pip by specifiying path to <path_to_repo>/seq2atac/
Some of the recurring scripts and notebooks are listed here for ease of navigation:

Notebook for lsi projection, nearest neighbors, differential enhancer analysis is in figure1/TCGA Differential to nearest neighbor Code.ipynb
Code for training models is inside figure2/pipeline_250_1364_prejitter/
Code for ISM-based motif/mutation scoring is in figure2/ism_motif_enrichment/
