#$ -q gpu-a100
#$ -cwd 
#$ -l h_vmem=40g
#$ -l h_rt=1:00:00 

conda activate python38
python /illumina/scratch/deep_learning/akumar22/seq2atac/seq2atac/stable/mutation_scoring_pipeline/mutation_scoring.py $mutations_file $model_type $model_file $outfile
