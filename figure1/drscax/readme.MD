These are helper codes for preparing data and training the denoising autoencoders.

create_sparsematrix.R is the R code that reads in all the peak by cell rds files obtained from ArchR and converts them into sparse matrices

create_adata.py is the python code that produces the log tf-idf transformer cell by peak count matrices as a scanpy h5ad object 

train.py, models.py, eval_.py and data.py are helper functions for training denoising autoencoders
