from datetime import datetime
import os, sys
import torchvision
from utils import append_evaluation_result, get_missing_run_parameters, update_eval_result, load_expl, arg_parse, getresultslist
import json
import time

## import from road module
import road
from road import run_road
from road.imputations import *
from road.retraining import *
import pudb
from termcolor import colored
# pudb.set_trace()
# different seeds
seeds = [2005, 42, 1515, 3333, 420]

# res_acc, prob_acc = run_road(model, dataset_test, expl_test, normalize_transform, [perc_value], morf=morf, batch_size=32, imputation = imputer)
for_rank_correlation = {"percentages": ps, 
 "base_methods": group, 
 "modifiers":modifiers, 
 "dataset": "imagenet", 
 "imputations": [imputation], 
 "orders": ["morf" if morf else "lerf"], 
 imputation: {group: {modifier: {("morf" if morf else "lerf"): {f"{perc_value}": {'test_probs':more_road_returns['test_probs'],
                'test_predictions':more_road_returns['test_predictions']} }}}}}        
#update_eval_result(res_acc[0].item(), storage_file, imputation, group, modifier, morf, perc_value, run_id)
value = res_acc[0].item()
res_dict, mylist = getresultslist(storage_file, imputation, group, modifier, morf, perc_value)
mylist[run_id] = value
#json.dump(res_dict, open(filepath, "w"))
