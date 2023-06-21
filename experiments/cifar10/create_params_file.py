import os
import json
import argparse
'''
{"basemethod": "ig",
 "modifiers": ["base", "sg", "sq", "var"],
 "imputation": "linear",
 "morf": true,
 "datafile": "result/noretrain.json",
 "percentages": [0.1,0.2,0.3,0.4,0.5,0.7,0.8,0.9],
 "timeoutdays": 0
}
'''
PARAMS_FOLDER = 'params'
parser = argparse.ArgumentParser()
parser.add_argument('--basemethod')
parser.add_argument('--modifiers',nargs='*')
parser.add_argument('--imputation')
parser.add_argument('--morf',type=lambda t:t.lower() == 'true')
parser.add_argument('--datafile')
parser.add_argument('--percentages',nargs='*',type=float)
parser.add_argument('--timeoutdays',type=int)
parser.add_argument('--retrain',type=lambda t:t.lower() == 'true')
parser.add_argument('--params_filename')
args = parser.parse_args()
print(args)
payload = {"basemethod": args.basemethod,
 "modifiers": args.modifiers,
 "imputation": args.imputation,
 "morf": args.morf,
 "datafile": args.datafile,
 "percentages": args.percentages,
 "timeoutdays": args.timeoutdays,
}

#filename = os.path.join(PARAMS_FOLDER,f'{"retrain" if args.retrain else "noretrain"}_{payload["basemethod"]}_{payload["imputation"]}_{"morf" if payload["morf"] else "lerf"}.json')
filename=args.params_filename
print(filename)
#import sys;sys.exit()
with open(filename,'w') as f:
    json.dump(payload,f)
