#!/bin/bash
export HOME=/tmp/charlieh

python /jet/home/houc/ptranking/e2e_eval.py -cuda 0 -dir_json $2 \
-pretrain_lr 0 -finetune_lr 0 -trial_num $3 -aug_type 0 -aug_percent 0 -dim $4 -layers $5 -temperature 0 -pretrainer LightGBMLambdaMART -mix 0 -shrink $1 -blend 0 -scale 0 -gumbel 0 \
-num_negatives 0 -freeze 0 -probe_layers 0 -finetune_only 0 -finetune_trials 0
