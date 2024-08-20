#$ -q gpu
#$ -cwd 
#$ -l mem=40g
#$ -l h_rt=00:30:00 

conda activate seq2atac
python ./mutation_scoring.py $mutations_file $model_type $model_file $outfile
