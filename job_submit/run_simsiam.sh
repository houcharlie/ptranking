#!/bin/bash

# make sure the script will use your Python installation, 
# and the working directory as its home location
#export PYTHONPATH=/afs/ece/usr/charlieh/.local/lib/python3.6
export HOME=/tmp/charlieh

# run your script
/afs/ece.cmu.edu/usr/charlieh/ptrank_venv/bin/python /afs/ece.cmu.edu/usr/charlieh/ptranking/simsiam_e2e.py -cuda 0 -dir_json /afs/ece.cmu.edu/usr/charlieh/ptranking/job_submit/inputs \
-pretrain_lr 0.00001 -finetune_lr $1 -trial_num $4 -aug_type $3 -aug_percent $2
