#$ -cwd 
#$ -l h_vmem=250g 

conda activate python38
python /illumina/scratch/deep_learning/akumar22/seq2atac/seq2atac/stable/somatic_mutation_pipeline/mutation_process.py $somatic_pkl $peakfile $outfile1 $outfile2
