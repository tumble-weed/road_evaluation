
# Call this script from the experiments/cifar10 folder as cwd.
# PYTHON="python -m ipdb -c c"
import subprocess
import os
from argparse import Namespace
import itertools
import time
import sys
import torch
sys.path.append('../../')
PYTHON="python"
GROADDIR="/root/evaluate-saliency-4/GPNN_for_road"
NO_RETRAIN_SCRIPT = "NoRetrainingMethod.py"
RETRAIN_SCRIPT = "RetrainingMethod.py"


#============================================================
# Arguments
#============================================================
"""
args = Namespace()
args.retrain=False
args.purge=False
"""
#======================================================
# Flags
#======================================================
env = dict(
            #==========================================
            # to only try high percent (high removal or high keep?)
            # export DBG_HIGH_PERCENT=1, 
            #==========================================
            # to only try with low percent
            # export DBG_LOW_PERCENT=1,
            #==========================================
            # to break out early from road_eval
            DBG_ROAD_EVAL_LOOP_BREAK=1,
            #==========================================
            # to break out early from retraining
            # DBG_BREAK_ROAD_TRAIN=1,
            #==========================================
            # to break early from percentages
            DBG_EARLY_BREAK_IN_RUN_ROAD=1,
            #==========================================
            # to break near end in gpnn-eval
            # DBG_BREAK_NEAR_END=1,
            #==========================================
            # use multiple workers for dataloader
            USE_MULTIPLE_DATALOADER_WORKERS=0,
            #==========================================
            # correct the issue in road_eval
            CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS=1,
            #==========================================
            purge = True
)
#=======================================================
# stubs related to flags
#=======================================================
road_eval_correction_stub= "_corrected_eval" if env['CORRECT_ROAD_EVAL_DATALOADER_PREDICTIONS'] else ""
flag_stub=f"{road_eval_correction_stub}"
print( road_eval_correction_stub)
flag_stub = road_eval_correction_stub

def get_create_params_filename(
    
    flag_stub = "",
    **kwargs):
    assert False,'deprecated'
    if  kwargs['retrain']:
        DATAFILE="result/retrain{flag_stub}.json"
        retrain_stub="retrain"
    else:    
        DATAFILE="result/noretrain{flag_stub}.json"
        retrain_stub="noretrain"
    if  kwargs['morf']:
        morf_stub="morf"
    else:
        morf_stub="lerf"
        params_filename=f"{PARAMS_FOLDER}/{retrain_stub}_{basemethod}_{imputation}_{morf_stub}{flag_stub}.json"
    if env.get('DONT_CREATE_PARAMS',False):
        print('skipping creation of params jsons')
        time.sleep(3)
    else:
        import ipdb;ipdb.set_trace()
        subprocess.run([PYTHON,"create_params_file.py","--basemethod", basemethod,
                         "--modifiers", MODIFIERS,"--imputation",imputation,"--morf", env['morf'],"--datafile", DATAFILE, "--percentages", PERCENTAGES, "--timeoutdays", TIMEOUTDAYS, "--retrain", env['retrain'] ,"--params_filename" ,params_filename])
    return params_filename


#=======================================================
# Actual run
#=======================================================
basemethods=["ig", "gb"]
MODIFIERS=["base" ,"sg" ,"sq" ,"var"]
imputations=["linear" ,"fixed" ,"gain"]
use_morfs = [True ,False]
retrains = [True,   False]
PERCENTAGES=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]

TIMEOUTDAYS=0
#DATAFILE="result/noretrain.json"
#retrains=(True False)
PARAMS_FOLDER="params"

param_gen = itertools.product(basemethods,imputations,use_morfs,retrains)
# for basemethod in basemethods:
#     for imputation in imputations:
#         for morf in use_morf:
    
for params in param_gen:
    basemethod, imputation, morf,retrain = params
    print(f'{basemethod} {imputation} {morf} {retrain}')
    """    
    params_filename = get_create_params_filename(flag_stub,
    retain = env["retrain"],
    morf = morf,
    )
    print(params_filename)
    """
    #=======================================================
    # Purge
    #=======================================================
    if env.get('SLOW',False):
        if env.get('purge',False):
            print('going to purge in 5 seconds')
        else:
            print("continuing previous results in 5 seconds")

    if  env.get('purge',False):
        if  retrain:
            print('purging retrain') 
            subprocess.run([PYTHON, "utils.py",f'--result_file=./result/retrain{flag_stub}.json'])
            # import ipdb;ipdb.set_trace()
            subprocess.run([PYTHON, "utils.py",f'--result_file=./result/retrain{flag_stub}_details.json'])
        else:
            print('purging noretrain')
            subprocess.run([PYTHON, "utils.py" ,f'--result_file=./result/noretrain{flag_stub}.json'])
            subprocess.run([PYTHON, "utils.py" ,f'--result_file=./result/noretrain{flag_stub}_details.json'])    
    #=======================================================
    
    if env.get("DRY_RUN",False):
        sys.exit() 
    if not retrain:
        SCRIPT = NO_RETRAIN_SCRIPT
        import NoRetrainingMethod
        oneRun = NoRetrainingMethod.oneRun
    else:
        import RetrainingMethod
        # from  RetrainingMethod import oneRun
        oneRun = RetrainingMethod.oneRun
        SCRIPT = RETRAIN_SCRIPT
    # import ipdb;ipdb.set_trace()
    """
    subprocess.run([PYTHON, SCRIPT,
        "--data_path=${GROADDIR}/road_evaluation/data", 
        "--expl_path=${GROADDIR}/road_evaluation/data", 
        f'--params_file=f"{params_filename}"',
        "--model_path='../../data/cifar_8014.pth'"])
    """
    # from RetrainingMethod import one
    data_path = f"{GROADDIR}/road_evaluation/data"
    expl_path = f"{GROADDIR}/road_evaluation/data"
    batch_size=32
    dataset = "cifar"
    use_device = "cuda" if torch.cuda.is_available() else "cpu"
    group = basemethod
    storage_file = f"result/{'retrain' if retrain else 'noretrain'}.json"
    storage_file_details = f"result/{'retrain' if retrain else 'noretrain'}_details.json"
    # storage_file_details = 
    modifiers = ["base","sg","sq","var"]
    ps = [0.1,0.3,0.5,0.6,0.7,0.8,0.9]
    timeout = 0
    epoch=1
    oneRun(
        data_path,
        expl_path,
        use_device ,
        batch_size,
        # params_file = args.params_file
        dataset,
        imputation,group,morf,
        storage_file,
        storage_file_details,modifiers,ps,
        timeout,
        epoch=epoch)

    # if True:
    #     if  not env['retrain']:
    #         subprocess.run([PYTHON, NO_RETRAIN_SCRIPT,
    #     "--data_path=${GROADDIR}/road_evaluation/data", 
    #     "--expl_path=${GROADDIR}/road_evaluation/data", 
    #     f'--params_file=f"{params_filename}"',
    #     "--model_path='../../data/cifar_8014.pth'"])
    #     # sys.exit()
    #     else
    #         subprocess.run([PYTHON,RETRAIN_SCRIPT 
    #     "--data_path=${GROADDIR}/road_evaluation/data 
    #     "--expl_path=${GROADDIR}/road_evaluation/data 
    #     "--params_file=f"{params_filename}"])


# params_files=("noretrain_params.json" "noretrain_params_gb.json" "noretrain_params_lerf.json" "noretrain_params_gb_lerf.json")
# for params_file in ${params_files}:
# DBG_ROAD_EVAL_LOOP_BREAK=1 python -m ipdb -c c NoRetrainingMethod.py \
#         --data_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
# 		--expl_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
# 		--params_file='$(params_file)' \
# 		--model_path='../../data/cifar_8014.pth'
# 
