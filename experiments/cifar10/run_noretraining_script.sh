#!/bin/bash
# Call this script from the experiments/cifar10 folder as cwd.
export PYTHONPATH=../../
#======================================================
# to only try high percent (high removal or high keep?)
export DBG_HIGH_PERCENT=1 
# to only try with low percent
# export DBG_LOW_PERCENT=1
# to break out early from road_eval
export DBG_ROAD_EVAL_LOOP_BREAK=1
# to break out early from retraining
# DBG_BREAK_ROAD_TRAIN=1
# to break early from percentages
# export DBG_EARLY_BREAK_IN_RUN_ROAD=1
# to break near end in gpnn-eval
# DBG_BREAK_NEAR_END=1
# use multiple workers for dataloader
export USE_MULTIPLE_DATALOADER_WORKERS=1
# correct the issue in road_eval
export CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS=1

#======================================================
python -m ipdb -c c NoRetrainingMethod.py \
        --data_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
		--expl_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
		--params_file='noretrain_params.json' \
		--model_path='../../data/cifar_8014.pth'


# params_files=("noretrain_params.json" "noretrain_params_gb.json" "noretrain_params_lerf.json" "noretrain_params_gb_lerf.json")
# for params_file in ${params_files[@]}; do
# DBG_ROAD_EVAL_LOOP_BREAK=1 python -m ipdb -c c NoRetrainingMethod.py \
#         --data_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
# 		--expl_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
# 		--params_file='$(params_file)' \
# 		--model_path='../../data/cifar_8014.pth'
# done
