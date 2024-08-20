JOB_NAME=$1
DURATION=""
if [ $3 = "main" ]; then
   DURATION=72
   queue_name=slice ## TODO: Correct CPU queue name
else
  DURATION=10
  queue_name=gpu ## TODO: Correct GPU queue name
fi

if [ ! -d "qsub_logs" ]; then
        mkdir qsub_logs
fi


qsub <<CMD
#!/bin/bash
#$ -N "$JOB_NAME"
#$ -cwd
#$ -j y
#$ -o qsub_logs/
#$ -e qsub_logs/
#$ -S /bin/bash
#$ -V
#$ -p 0
#$ -q ${queue_name}
#$ -l mem=64g,h_rt=${DURATION}:00:00

echo [$(date)] ${JOB_NAME} STARTED 

conda activate seq2atac
module load cuda11.2/toolkit/11.2.2 cudnn/8.2.4

python -u ./motifism_score.py --file_specifier $2 --program_switch $3 --file $4

echo [$(date)] ${JOB_NAME} DONE
CMD
