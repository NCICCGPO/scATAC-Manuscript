import os

exptroot="./models/" ### TODO: output path
peakroot="./peaks/" ### TODO: peaks must be of the form $peak_root/$name_peakset.csv

all_cancers = ["BLCA","BRCA","GBM","COAD","KIRC","KIRP","LUAD","SKCM"] ### cancer types
all_cancers += [f'BLCA{x}' for x in range(1,10)] + ['BLCA_control'] ### BLCA samples + control
all_cancers += [f"BRCA{x}" for x in range(10,26)] + ['BRCA_control'] ### BRCA samples + control
all_cancers += ["GBM45_cloneA","GBM45_cloneB","GBM39_cloneA","GBM39_cloneB"] ### GBM Subclones
all_cancers += ["CancerBcells", "CancerMacrophages", "CancerTcells", "PBMC_Bcells", "PBMC_Macrophages", "PBMC_Tcells"] ### Immune cells and control
all_cancers += ['LUAD55', 'LUAD56', 'LUAD57', 'LUAD59', 'LUAD60', 'LUAD61', 'LUAD62', 'LUAD63', 'LUAD64', 'LUAD65','LUAD_fetal_control','LUAD_fetal_adult','LUAD_fetal_adult'] ### LUAD samples + control
all_cancers += ["Breast_BASAL_healthy","Breast_LuminalHR_healthy","Breast_LuminalSEC_healthy"] ### Healthy breast samples
all_cancers += ["KIRC47","KIRC48","KIRC49","KIRC50","KIRP51","KIRP52","KIRP53","KIRP54"] ### KIRC samples

### Fold-wise scripts

script = f"./foldtrainer.sh"

gc_match_size = 250
model_input_size = 1364
model_type = f"conv_1364"

for cancer_name in all_cancers:
    for fold_number in range(5):
        peakfile = f"{peakroot}{cancer_name}_peakset.csv"
        assert os.path.exists(peakfile)
        outdir = f"{exptroot}/{cancer_name}/fold_{fold_number}"
        
        # create fold dir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            print("Created: ",outdir)
        master_file = f"{outdir}/master.csv"

        jobname = f"{cancer_name}_prejitter_fold{fold_number}"
        logsfile = f"{outdir}/{jobname}.txt"


        qsub_command = "qsub "+\
                       f"-v peakfile={peakfile} "+\
                       f"-v outdir={outdir} "+\
                       f"-v fold_number={fold_number} "+\
                       f"-v gc_match_size={gc_match_size} "+\
                       f"-v model_input_size={model_input_size} "+\
                       f"-v model_type={model_type} "+\
                       f"-N {jobname} -e {logsfile} -o {logsfile} {script}"

        print(qsub_command)
        print("")
        os.system(qsub_command)

