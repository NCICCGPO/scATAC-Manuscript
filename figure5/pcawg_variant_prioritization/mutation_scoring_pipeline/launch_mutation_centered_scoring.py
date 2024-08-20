import os

PATH_TO_MODELS = "./models/" ### TODO: organize $PATH_TO_MODELS/$cancer_name/fold_*/model.h5
exptroot="./scores/" ### TODO
mutations_root = "./variants/" ### TODO: must be $mutations_root/$cancer_name/variants_to_score.csv

all_cancers = ["BLCA","BRCA","GBM","COAD","KIRC","KIRP","LUAD","SKCM"] ### this can be any model
### Fold-wise scripts

script = f"./mutation_scoring.sh"
model_type = "conv_1364"

for c_idx,cancer_name in enumerate(all_cancers):
    mutations_file = f"{mutations_root}/{cancer_name}/variants_to_score.csv"
    for fold_num in range(5):
        model_file = f"{PATH_TO_MODELS}/{cancer_name}/fold_{fold_num}/model.h5"
        outfile =  f"{exptroot}/{cancer_name}/fold_{fold_num}_scored_mutation_centered.pkl"
        outdir =  f"{exptroot}/{cancer_name}/logs/"
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        jobname = f"{cancer_name}_fold{fold_num}_scoring_mutation_centered"
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
