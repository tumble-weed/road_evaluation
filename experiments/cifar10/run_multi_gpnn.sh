#!/bin/bash
# Call this script from the experiments/cifar10 folder as cwd.
# PYTHON="python -m ipdb -c c"
PYTHON="python"
export PYTHONPATH=../../
#============================================================
# Arguments
#============================================================
SLOW=true
retrain=false
purge=false
while [[ $# -gt 1 ]]; do
    case $1 in
        --retrain)
            echo "$2"
            if [[ "$2" == "true"  ]]; then
                retrain=true
            fi
            shift 2
            ;;
        --slow)
            echo "$2"
            if [[ "$2" == "false" ]]; then
                SLOW=false
            fi
            shift 2
            echo "SLOW:${SLOW}"
            ;;
        --purge)
            if [[ "$2" == "true"  ]]; then
                purge=true
            fi
            shift 2
            ;;
        *)
            echo "invalid option"
            shift
            ;;
    esac
done
echo "retrain:$retrain"
#======================================================
# Flags
#======================================================
# to only try high percent (high removal or high keep?)
# export DBG_HIGH_PERCENT=1 
# to only try with low percent
# export DBG_LOW_PERCENT=1
# to break out early from road_eval
# export DBG_ROAD_EVAL_LOOP_BREAK=1
# to break out early from retraining
# DBG_BREAK_ROAD_TRAIN=1
# to break early from percentages
# export DBG_EARLY_BREAK_IN_RUN_ROAD=1
# to break near end in gpnn-eval
# DBG_BREAK_NEAR_END=1
# use multiple workers for dataloader
export USE_MULTIPLE_DATALOADER_WORKERS=0
# correct the issue in road_eval
if ! test -v CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS; then
    export CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS=1
fi
echo $CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS
#exit 1
#=======================================================
# stubs related to flags
#=======================================================
road_eval_correction_stub=""
if [ "$CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS" -eq 1 ]; then
    road_eval_correction_stub="_corrected_eval"
    echo "correcting road_eval"
fi
flag_stub="${road_eval_correction_stub}"
# echo $flag_stub
echo $road_eval_correction_stub
# exit 1
#=======================================================
# Purge
#=======================================================
if [ $SLOW = true ]; then
    if [ $purge = true ]; then 
        echo "going to purge in 5 seconds"
    else
        echo "continuing previous results in 5 seconds"
    fi
    sleep 5
fi
if [ $purge = true ]; then
    if [ $retrain = true ]; then
        echo "purging retrain" 
        python utils.py --result_file="./result/retrain${flag_stub}_gpnn.json"
        python utils.py --result_file="./result/retrain${flag_stub}_gpnn_details.json"
    else
        echo "purging noretrain"
        python utils.py --result_file="./result/noretrain${flag_stub}_gpnn.json"
        python utils.py --result_file="./result/noretrain${flag_stub}_gpnn_details.json"
    fi
fi
#=======================================================
# Actual run
#=======================================================
basemethods=("ig" "gb")
MODIFIERS=("base" "sg" "sq" "var")
imputations=("gpnn")
use_morf=(true false)
PERCENTAGES=(0.1 0.2 0.3 0.4 0.5 0.7 0.8 0.9)
TIMEOUTDAYS=0
#DATAFILE="result/noretrain.json"
#retrains=(true false)
PARAMS_FOLDER="params"
for basemethod in ${basemethods[@]}; do
    for imputation in ${imputations[@]}; do
        for morf in ${use_morf[@]}; do
            #for retrain in ${retrains[@]}; do
                if [ $retrain = true ]; then
                    DATAFILE="result/retrain${flag_stub}_gpnn.json"
                    retrain_stub="retrain"
                else    
                    DATAFILE="result/noretrain${flag_stub}_gpnn.json"
                    retrain_stub="noretrain"
                fi
                echo "${basemethod} ${imputation} ${morf} ${retrain}"
                if [ $morf = true ]; then
                    morf_stub="morf"
                else
                    morf_stub="lerf"
                fi
                
                params_filename="${PARAMS_FOLDER}/${retrain_stub}_${basemethod}_${imputation}_${morf_stub}${flag_stub}.json"
                if ! test -v DONT_CREATE_PARAMS; then
                    python create_params_file.py --basemethod ${basemethod} --modifiers ${MODIFIERS[@]} --imputation $imputation --morf $morf --datafile $DATAFILE --percentages ${PERCENTAGES[@]} --timeoutdays $TIMEOUTDAYS --retrain $retrain --params_filename $params_filename
                else
                    echo "skipping creation of params jsons"
                    sleep 3
                fi
                echo $params_filename
                if test -v DRY_RUN; then
                    exit 0 
                fi
                if [ $retrain = false ]; then
                    $PYTHON NoRetrainingMethod.py \
                --data_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
                --expl_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
                --params_file="${params_filename}" \
                --model_path='../../data/cifar_8014.pth'
                else
                    $PYTHON RetrainingMethod.py \
                --data_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
                --expl_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
        --params_file="${params_filename}"
                fi
            #done
        done

    done

done
exit 0    



# params_files=("noretrain_params.json" "noretrain_params_gb.json" "noretrain_params_lerf.json" "noretrain_params_gb_lerf.json")
# for params_file in ${params_files[@]}; do
# DBG_ROAD_EVAL_LOOP_BREAK=1 python -m ipdb -c c NoRetrainingMethod.py \
#         --data_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
# 		--expl_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
# 		--params_file='$(params_file)' \
# 		--model_path='../../data/cifar_8014.pth'
# done
