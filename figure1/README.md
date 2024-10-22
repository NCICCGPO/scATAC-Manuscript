This folder contains all the scripts required for training and evaluating the Denoising Autoencoder used in Figure 1

The notebook "DenoisingAutoEncoder Training.ipynb" contains the code for training the denoising autoencoder model that takes in as the input the cell by peak (top 50k log TF-IDF transformed matrix) and predicts CNV normalized cell by peak matrix as output. The training uses samples with CNV calls from the 80X WGS made available in the study. The input to the model is made available as an h5ad object "tcga_canceronly_top50klogtfidf_221011.h5ad" as part of the publication page, and the trained model weight used for generating the denoised embeddings is available as a torch model dictionary "v062_woimmune_bst8layer50k_221012_165814.pt" as part of the publication page.

The notebook "Denoising AutoEncoder - Evaluation.ipynb" contains the code for generating the denoised embeddings for all tumor cells in the study. The trained model weight "tcga_canceronly_top50klogtfidf_221011.h5ad" is available on the publication page. The predictions are available for all tumor cells identified in Table S1. The notebook also generates the "Z_PCA.npy" file with denoised embeddings in the nearest neighbor analysis.

The notebook "Denoising AutoEncoder Nearest Neighbor Analysis.ipynb" contains code for comparing the numbers of cells of the same sample vs. different samples of the same cancer type analysis using the denoised embeddings. The "Z_PCA.npy" is generated as part of the "Denoising AutoEncoder - Evaluation.ipynb" notebook. The random cells chosen for the analysis are available on the publication page.

The notebook "IterativeLSI Nearest Neighbor Analysis.ipynb" contains code for comparing the numbers of cells of the same sample vs. different samples of the same cancer type analysis using the LSI embeddings from IterativeLSI. The "matSVD_tumor.csv" with IterativeLSI embeddings for all tumor cells is available on the publication page. 

The notebook "TCGA Immune Cells Differential to nearest neighbor Code.ipynb" contains code for the immune cell types projection analysis. The Archr objects of the cancer, healthy, and combined immune projects are available on the publication page.

The notebook "TCGA-Computing the 10MB  CNV like ATAC signal.ipynb" contains code for adding the CNV matrix (10MB atac CNV matrix) to archr project. The file TCGA_Helper.R contains helper functions for making this matrix.  
