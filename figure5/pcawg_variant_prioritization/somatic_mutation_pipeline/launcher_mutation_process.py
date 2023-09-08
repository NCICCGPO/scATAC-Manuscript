import os

peakroot="/illumina/scratch/deep_learning/akumar22/TCGA/mutations_scoring/master_files/"
mutations_root="/illumina/scratch/deep_learning/akumar22/TCGA/mutation_prioritization/cancer_specific_somatic_hg38/"
outdir1="/illumina/scratch/deep_learning/akumar22/TCGA/mutations/somatic_filtered/"
outdir2="/illumina/scratch/deep_learning/akumar22/TCGA/mutations/somatic_filtered_regulatory/"
if not os.path.exists(outdir1):
    os.makedirs(outdir1)

if not os.path.exists(outdir2):
    os.makedirs(outdir2)

all_cancers = ['COAD', 'SKCM', 'BRCA', 'LUAD', 'GBM', 'BLCA', 'KIRC', 'KIRP']
### Fold-wise scripts

script = "/illumina/scratch/deep_learning/akumar22/seq2atac/seq2atac/stable/somatic_mutation_pipeline/mutation_process.sh"


for cancer_name in all_cancers:

    somatic_pkl = f"{mutations_root}/{cancer_name}_somatic_hg38.pkl"
    peakfile = f"{peakroot}/cancer_peaks_500/{cancer_name}_peaks_indexed.bed"
    outfile1 = f"{outdir1}/{cancer_name}_filtered_annotated_somatic.pkl"
    outfile2 = f"{outdir1}/{cancer_name}_filtered_annotated_somatic_regulatory.pkl"

    logdir = outdir1 + "/logs/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    jobname = f"somatic_filter_{cancer_name}"
    logsfile = f"{logdir}/{jobname}.txt"


    qsub_command = "qsub "+\
                    f"-v somatic_pkl={somatic_pkl} "+\
                    f"-v peakfile={peakfile} "+\
                    f"-v outfile1={outfile1} "+\
                    f"-v outfile2={outfile2} "+\
                    f"-N {jobname} -e {logsfile} -o {logsfile} {script}"
    print(qsub_command)
    print("")
    os.system(qsub_command)
