#!/bin/bash

python /jet/home/houc/ptranking/e2e_eval.py -cuda 0 -dir_json /jet/home/houc/ptranking/job_submit/inputs \
-pretrain_lr $1 -finetune_lr $2 -trial_num $3 -aug_type $4 -aug_percent $5 -dim $6 -layers $7 -temperature $8 -pretrainer $9 -mix ${10} -shrink ${11} -blend ${12} -scale ${13} -gumbel ${14}
