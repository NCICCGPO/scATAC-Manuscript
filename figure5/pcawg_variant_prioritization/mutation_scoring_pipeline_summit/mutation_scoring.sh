#$ -q gpu-a100
#$ -cwd 
#$ -l mem=40g
#$ -l h_rt=00:30:00 

conda activate python38
python /illumina/scratch/deep_learning/akumar22/seq2atac/seq2atac/stable/mutation_scoring_pipeline_summit/mutation_scoring.py $mutations_file $model_type $model_file $outfile
