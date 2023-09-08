import os
mutations_root="/illumina/scratch/deep_learning/akumar22/TCGA/mutations_scoring/atac_seq_samples/somatic_filtered_regulatory/"
exptroot="/illumina/scratch/deep_learning/akumar22/TCGA/mutations_scoring/atac_seq_samples/somatic_filtered_regulatory/"
all_cancers = ["BLCA","BRCA","GBM","COAD","KIRC","KIRP","LUAD","SKCM"]


### Fold-wise scripts

script = "/illumina/scratch/deep_learning/akumar22/seq2atac/seq2atac/stable/mutation_scoring_pipeline/mutation_scoring.sh"
model_type = "conv_1364"

for c_idx,cancer_name in enumerate(all_cancers):
    mutations_file = f"{mutations_root}/{cancer_name}_filtered_regulatory_somatic.csv"
    for fold_num in range(5):
        model_file = f"/illumina/scratch/deep_learning/akumar22/TCGA/models_250_1364_minibatch_prejitter/{cancer_name}/fold_{fold_num}/model.h5"
        outfile =  f"{exptroot}/{cancer_name}/fold_{fold_num}_scatac_mutation_centered.csv"
        outdir =  f"{exptroot}/{cancer_name}/logs/"
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        jobname = f"{cancer_name}_fold{fold_num}_scatac_mutation_centered"
        logsfile = f"{outdir}/{jobname}.txt"

        qsub_command = "qsub "+\
                        f"-v mutations_file={mutations_file} "+\
                        f"-v model_type={model_type} "+\
                        f"-v model_file={model_file} "+\
                        f"-v outfile={outfile} "+\
                        f"-N {jobname} -e {logsfile} -o {logsfile} {script}"


        print(qsub_command)
        print("")
        os.system(qsub_command)
