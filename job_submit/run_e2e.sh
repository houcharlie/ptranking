#!/bin/bash

# make sure the script will use your Python installation, 
# and the working directory as its home location
#export PYTHONPATH=/afs/ece/usr/charlieh/.local/lib/python3.6
export HOME=/tmp/charlieh

# run your script
/afs/ece.cmu.edu/usr/charlieh/ptrank_venv/bin/python /afs/ece.cmu.edu/usr/charlieh/ptranking/e2e_eval.py -cuda 0 -dir_json /afs/ece.cmu.edu/usr/charlieh/ptranking/job_submit/inputs \
-pretrain_lr $1 -finetune_lr $2 -trial_num $3 -aug_type $4 -aug_percent $5 -dim $6 -layers $7 -temperature $8 -pretrainer $9
