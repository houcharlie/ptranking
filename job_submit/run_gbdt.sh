#!/bin/bash

python /home/ubuntu/ptranking/e2e_eval.py -cuda 0 -dir_json /home/ubuntu/ptranking/testing/ltr_tree/json/ \
-pretrain_lr 0 -finetune_lr 0 -trial_num 0 -aug_type 0 -aug_percent 0 -dim 0 -layers 0 -temperature 0 -pretrainer LightGBMLambdaMART -mix 0 -shrink $1 -blend 0 -scale 0 -gumbel 0
