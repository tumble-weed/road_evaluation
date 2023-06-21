#!/bin/bash
# Call this script from the experiments/cifar10 folder as cwd.
export PYTHONPATH=../../
# export DBG_BREAK_ROAD_EVAL="1"

# CUDA_VISIBLE_DEVICES=, python -m ipdb -c c NoRetrainingMethod.py \
#BREAK_NEAR_END=1 CUDA_VISIBLE_DEVICES=, python -m ipdb -c c NoRetrainingMethod.py \
#CUDA_VISIBLE_DEVICES=, python -m ipdb -c c NoRetrainingMethod.py \
# BREAK_NEAR_END=1 python -m ipdb -c c NoRetrainingMethod.py \
#IMPORT_NEW_GPNN=1 python -m ipdb -c c NoRetrainingMethod_imagenet.py \
IMPORT_NEW_GPNN=1 pudb NoRetrainingMethod_imagenet.py \
        --data_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
		--expl_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
		--params_file='noretrain_params_imagenet.json' \
		--model_path='../../data/cifar_8014.pth'
