#$ -q gpu-a100
#$ -cwd 
#$ -l h_vmem=200g 
#$ -l h_rt=24:00:00 

conda activate python38
python /illumina/scratch/deep_learning/akumar22/seq2atac/seq2atac/stable/pipeline_250_1364_prejitter/foldtrainer.py $peakfile $outdir $fold_number $gc_match_size $model_input_size $model_type
