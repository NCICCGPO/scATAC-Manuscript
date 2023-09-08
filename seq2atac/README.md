Seq2ATAC repo

Illumina Confidential - do not distribute!

seq2atac/ - defines data pipeline, models, training pipeline and some analysis functions for TCGA project
expt_scripts/ - create an experiment using functions from seq2atac. Also contains scripts to submit jobs to SGE
tests/ - currently empty, could be used to add test cases to ensure data sanity

Installation instructions
- please into the repo directory in your dev machine
- create virtual conda environment with Python3.8/9
- install requirements.txt using `pip install -r requirements.txt`
- execute `pip install -e .`
