#!/bin/bash
# Call this script from the experiments/cifar10 folder as cwd.
export PYTHONPATH=../../
# Call this script from the experiments/cifar10 folder as cwd.
export PYTHONPATH=../../
# to only try high percent (high removal or high keep?)
export DBG_HIGH_PERCENT=1 
# to only try with low percent
# export DBG_LOW_PERCENT=1
# to break out early from road_eval
export DBG_ROAD_EVAL_LOOP_BREAK=1
# to break early from percentages
# export DBG_BREAK_ROAD_EVAL=1
# to break near end in gpnn-eval
# DBG_BREAK_NEAR_END=1

python -m ipdb -c c RetrainingMethod.py \
        --data_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
		--expl_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
		--params_file='retrain_params.json'

