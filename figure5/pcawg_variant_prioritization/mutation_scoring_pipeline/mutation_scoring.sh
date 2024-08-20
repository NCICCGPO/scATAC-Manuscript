#$ -q gpu
#$ -cwd 
#$ -l h_vmem=40g

conda activate seq2atac
python ./mutation_scoring.py $mutations_file $model_type $model_file $outfile
