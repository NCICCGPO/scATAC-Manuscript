import os
mutations_root="/illumina/scratch/deep_learning/akumar22/TCGA/mutations_scoring/gwas_breast_control/"
outdir =  f"{mutations_root}/logs/"
if not os.path.exists(outdir):
    os.makedirs(outdir)
# all_cancers = ["BLCA","BRCA","GBM","COAD","KIRC","KIRP","LUAD","SKCM"]

### Fold-wise scripts

script = "/illumina/scratch/deep_learning/akumar22/seq2atac/seq2atac/stable/mutation_scoring_pipeline_summit/mutation_scoring.sh"
model_type = "conv_1364"

mutations_file = f"{mutations_root}/breast_cancer_gwas_ld_expanded.pkl"
model_file_pattern = "/illumina/scratch/deep_learning/akumar22/TCGA/models_250_1364_minibatch_prejitter/BRCA/fold_{0}/model.h5"
outfile_pattern =  mutations_root + "/BRCA/fold_{0}_gwas_summit_centered_ld_expanded.pkl"
jobname_pattern = "BRCA_fold{0}_gwas_summit_centered_ld_expanded"


for fold_num in range(5):
    model_file = model_file_pattern.format(fold_num)
    outfile =  outfile_pattern.format(fold_num)
    jobname = jobname_pattern.format(fold_num)
    logsfile = f"{outdir}/{jobname}.txt"

    tomakedir,_ = os.path.split(outfile)
    if not os.path.exists(tomakedir):
        os.makedirs(tomakedir)

    qsub_command = "qsub "+\
                    f"-v mutations_file={mutations_file} "+\
                    f"-v model_type={model_type} "+\
                    f"-v model_file={model_file} "+\
                    f"-v outfile={outfile} "+\
                    f"-N {jobname} -e {logsfile} -o {logsfile} {script}"


    print(qsub_command)
    print("")
    os.system(qsub_command)
