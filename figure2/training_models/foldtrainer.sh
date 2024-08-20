#$ -q gpu
#$ -cwd 
#$ -l h_vmem=200g 
#$ -l h_rt=24:00:00 

conda activate seq2atac
python ./foldtrainer.py $peakfile $outdir $fold_number $gc_match_size $model_input_size $model_type
