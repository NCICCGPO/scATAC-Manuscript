import os

exptroot="/illumina/scratch/deep_learning/akumar22/TCGA/models_250_1364_minibatch_prejitter/breast_control_consensus/"

### Fold-wise scripts

script = "/illumina/scratch/deep_learning/akumar22/seq2atac/seq2atac/stable/pipeline_250_1364_prejitter/foldtrainer.sh"

gc_match_size = 250
model_input_size = 1364
model_type = f"conv_1364"

for fold_number in range(5):
    peakfile = f"/illumina/scratch/deep_learning/lsundaram/singlecelldatasets/TCGA/ArchR_Projects/controls/pan_control_BREAST_NEW_onlytissue/pancontrol_breast_relevant_peaks.csv"
    outdir = f"{exptroot}/fold_{fold_number}"

    # create fold dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print("Created: ",outdir)
    master_file = f"{outdir}/master.csv"

    jobname = f"consensus_brca_control_prejitter_fold{fold_number}"
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

