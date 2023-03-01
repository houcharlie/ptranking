#!/bin/bash

# make sure the script will use your Python installation, 
# and the working directory as its home location
#export PYTHONPATH=/afs/ece/usr/charlieh/.local/lib/python3.6
export HOME=/tmp/charlieh

# run your script
/afs/ece.cmu.edu/usr/charlieh/ptrank_venv/bin/python /afs/ece.cmu.edu/usr/charlieh/ptranking/e2e_eval.py -cuda 0 -dir_json /afs/ece.cmu.edu/usr/charlieh/ptranking/testing/ltr_tree/json/ \
-pretrain_lr 0 -finetune_lr 0 -trial_num 0 -aug_type 0 -aug_percent 0 -dim 0 -layers 0 -temperature 0 -pretrainer LightGBMLambdaMART -mix 0 -shrink $1 -blend 0 -scale 0 -gumbel 0
