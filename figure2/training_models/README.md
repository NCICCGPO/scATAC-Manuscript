This is the folder for training all models. For each peakset, we can train 5 fold models on a compute cluster that supports qsub.

launch_training_jobs.py is the main entry point to be run. 
Please set the following variables
PATH_TO_REPO: path to this codebase
exptroot: where the models and negative sampled regions would come
peakroot: path to all peaks. Note that peaks must be of the form $peak_root/$name_peakset.csv, where name is the name of the peak (example: BRCA, BLCA, LUAD55)
Please set the following variables in foldtrainer.py
fasta_file: path to hg38 fasta
windowed_1364_gc_file: windowed genome file with gc content pre-computed to 2 precision 

This requires seq2atac to be installed and the seq2atac environment to be installed (see seq2atac folder)
