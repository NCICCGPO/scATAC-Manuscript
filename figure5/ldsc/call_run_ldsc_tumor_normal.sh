tissues=(BREAST COLON BRAIN BLADDER KIDNEY)
tts=(BRCA COAD GBM BLCA KIRC)

for i in 0 2;
# for i in {1..4} ;
    do
        # suffix=tumor; qsub -cwd -b Y -l h_vmem=50G -N ${tts[$i]}_${suffix} "conda activate ldsc; bash /illumina/scratch/deep_learning/asalcedo/scATAC/code/run_ldsc_all_tumor_normal_1.sh ${tts[$i]} $suffix"
        suffix=normal; qsub -cwd -b Y -l h_vmem=50G -N ${tts[$i]}_${suffix} "conda activate ldsc; bash /illumina/scratch/deep_learning/asalcedo/scATAC/code/run_ldsc_all_tumor_normal_1.sh ${tissues[$i]} $suffix"
        # suffix=tumorNN; qsub -cwd -b Y -l h_vmem=50G -N ${tts[$i]}_${suffix} "conda activate ldsc; bash /illumina/scratch/deep_learning/asalcedo/scATAC/code/run_ldsc_all_tumor_normal_1.sh ${tissues[$i]} $suffix"
        # suffix=tumor_non_normal; qsub -cwd -b Y -l h_vmem=50G -N ${tts[$i]}_${suffix} "conda activate ldsc; bash /illumina/scratch/deep_learning/asalcedo/scATAC/code/run_ldsc_all_tumor_normal_1.sh ${tts[$i]} $suffix"
        # suffix=NNnormal_tumor_overlap; qsub -cwd -b Y -l h_vmem=50G -N ${tts[$i]}_${suffix} "conda activate ldsc; bash /illumina/scratch/deep_learning/asalcedo/scATAC/code/run_ldsc_all_tumor_normal_1.sh ${tts[$i]}_${tissues[$i]} $suffix"
        # suffix=NNnormal_nontumor; qsub -cwd -b Y -l h_vmem=50G -N ${tts}_${suffix} "conda activate ldsc; bash /illumina/scratch/deep_learning/asalcedo/scATAC/code/run_ldsc_all_tumor_normal_1.sh ${tissues[$i]} $suffix"
    done
