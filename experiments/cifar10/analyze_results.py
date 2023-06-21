'''
set of utilities for analyzing the saved results jsons from
road evaluation experiments

TODO:
- we can analyze json-wise, where each json is either train/noretrain, imagenet/cifar, corrected/uncorrected.
- or we can query a particular experiment according to the config
'''
import json
import colorful
import itertools
import numpy as np    
def query(retrain=None, dataset=None, corrected=None):
    pass


# read_json to read a json file structure like above and return a dictionary organized as:
#    imputation:{}


def query_json(loaded,imputation,base_method,modifier,order,percentage):
    print('1',end='')
    if imputation not in loaded:
        return None
        pass
    print('2',end='')
    if base_method not in loaded[imputation]:
        return None
        pass
    print('3',end='')
    if modifier not in loaded[imputation][base_method]:
        return None
        pass         
    print('4',end='')                  
    if order not in loaded[imputation][base_method][modifier]:
        return None
        pass       
    print('5',end='')
    if isinstance(percentage,float):
        percentage = str(percentage)
    if percentage not in loaded[imputation][base_method][modifier][order]:
        # import ipdb;ipdb.set_trace()
        return None
        pass
    print('6',end='')
    # import ipdb;ipdb.set_trace()
    payload = loaded[imputation][base_method][modifier][order][percentage]
    # if isinstance(payload,dict):
    #     probs,predictions = payload['probs'],payload['predictions']
    #     return probs,predictions
    return payload

def get_keys(json_path):
    with open(json_path) as f:
        loaded = json.load(f)
    
    percentages = loaded['percentages']
    orders = loaded['orders']
    modifiers = loaded['modifiers']
    imputations = loaded['imputations']
    dataset  = loaded['dataset']
    base_methods = loaded['base_methods']
    return imputations,base_methods,modifiers,orders,percentages    
def read_json_and_flatten(json_path):
    #===========================================
    with open(json_path) as f:
        loaded = json.load(f)
    
    # percentages = loaded['percentages']
    # orders = loaded['orders']
    # modifiers = loaded['modifiers']
    # imputations = loaded['imputations']
    # dataset  = loaded['dataset']
    # base_methods = loaded['base_methods']
    imputations,base_methods,modifiers,orders,percentages  = get_keys(json_path)
    with open(json_path) as f:
        loaded = json.load(f)
    
    #===========================================
    to_return = {}
    #===========================================
    for imputation in imputations:
        for base_method in base_methods:
            for modifier in modifiers:
                for order in orders:
                    for percentage in percentages:
                        
                        out = query_json(loaded,imputation,base_method,modifier,order,percentage)                        
                        # import ipdb;ipdb.set_trace()
                        if out is not None:
                            # probs,predictions = out
                            # predictions = out['predictions']
                            # probs = out['probs']
                            
                            to_return[(imputation,base_method,modifier,order,percentage)] = out
    return to_return
#===================================================
results_folder = '/root/evaluate-saliency-4/GPNN_for_road/road_evaluation/experiments/cifar10/result'
import os
assert os.path.isdir(results_folder),'results folder does not exist'
json_path = os.path.join(results_folder, 'noretrain_details.json')
flat= read_json_and_flatten(json_path)
# def read_json_for_correlation(json_path,base_method,modifier):
#     flat= read_json_and_flatten(json_path)
#     flat = {k:v for k,v in flat.items() if k[1]==base_method and k[2]==modifier}
#     # pass
pass


def create_order_scores(flat):


    order_scores = {base_method:{order:{k:None for k in itertools.product(imputations,modifiers)} for order in orders} for base_method in base_methods}
    # for base_method in base_methods:
    #     for order in orders:
    #         for imputer_modifier in itertools.product(imputations,modifiers):
    #             for k in flat.keys():
    
    nsamples = None        
    for k in flat.keys():
        imputation,base_method,modifier,order,percentage = k
        if order_scores[base_method][order][imputation,modifier] is None:
            nruns =len(flat[k]['probs'])
            if nsamples is None:
                nsamples = len(flat[k]['probs'][0])
            else:
                assert len(flat[k]['probs'][0] ) == nsamples
            order_scores[base_method][order][imputation,modifier] = np.zeros((nruns,nsamples))
        order_scores[base_method][order][imputation,modifier] += np.array(flat[k]['probs'])
    return order_scores

def calculate_ranks(order_scores):
    ranks = {}
    for base_method in base_methods:
        ranks[base_method] = {}
        for imputation in imputations:
            ranks[base_method][imputation] = {}
            for order in orders:
                
                scores = [order_scores[base_method][order][imputation,modifier] for modifier in modifiers]

                if order == 'morf':
                    # for morf lower scores are better
                    scores = np.array(scores)
                    scores = scores.mean(axis = 1)
                    # import ipdb;ipdb.set_trace()
                    sort_order = np.argsort(scores,axis=0)
                    ranksi = np.argsort(sort_order,axis=0)
                    # import ipdb;ipdb.set_trace()
                elif order == 'lerf':
                    # for lerf higher scores are better
                    # sort in descending order
                    scores = np.array(scores)
                    scores = scores.mean(axis = 1)
                    sort_order = np.argsort((scores),axis=0)[::-1]
                    ranksi = np.argsort(sort_order,axis=0)
                    
                    # import ipdb;ipdb.set_trace()
                ranks[base_method][imputation][order] =  ranksi
    return ranks

if True:
    json_paths = {
                    # 'retrain':os.path.join(results_folder, 'retrain_details.json'),
                  'noretrain':os.path.join(results_folder, 'noretrain_details.json')
                  }
    
    train_flags = json_paths.keys()
    
    imputations,base_methods,modifiers,orders,percentages  = get_keys(json_paths['noretrain'])
    if 'retrain' in train_flags:
        imputations1,base_methods1,modifiers1,orders1,percentages1  = get_keys(json_paths['retrain'])
        assert all(
            [ imputations == imputations1, base_methods == base_methods1, modifiers == modifiers1, orders == orders1, percentages == percentages1]
        )
    
    nmodifiers = len(modifiers)
    nsamples = None
    ranks = {'retrain':{},'noretrain':{}}
    for train_flag in train_flags:
        order_scores = create_order_scores(flat)
        ranksi = calculate_ranks(order_scores)    
        ranks[train_flag]  = ranksi 
    corrs = {}
    for base_method in base_methods:
        corrs[base_method] = {}
        for imputation0,order0,train_flag0 in itertools.product(imputations,orders,train_flags):
            for imputation1,order1,train_flag1 in itertools.product(imputations,orders,train_flags):
                # print(imputation1,order1)
                # nruns0 = ranks[base_method][imputation0][order0].shape[1]
                # nruns1 = ranks[base_method][imputation1][order1].shape[1]
                
                assert ranks[train_flag0][base_method][imputation0][order0].ndim == 2
                if nsamples == None:
                    nsamples = ranks[train_flag0][base_method][imputation0][order0].shape[-1]
                assert ranks[train_flag0][base_method][imputation0][order0].shape == (nmodifiers,nsamples)
                
                corrs[base_method][
                    ((imputation0,order0,train_flag0),(imputation1,order1,train_flag1))] = None
                    
                
                R0 = ranks[train_flag0][base_method][imputation0][order0]
                R1 = ranks[train_flag1][base_method][imputation0][order1]
                cov_R0_R1 = np.mean(R0*R1,axis=0) - np.mean(R0,axis=0)*np.mean(R1,axis=0)
                sigma_R0 = np.std(R0,axis=0)
                sigma_R1 = np.std(R1,axis=0)
                assert cov_R0_R1.shape == (nsamples,)
                assert sigma_R0.shape == (nsamples,)
                assert sigma_R1.shape == (nsamples,)
                r = cov_R0_R1/(sigma_R0*sigma_R1)
                
                # import ipdb;ipdb.set_trace()
                if not os.environ.get('DBG_IGNORE_ALL_ONES',False) == '1':
                    if np.allclose(r,np.ones(r.shape)) :
                        import ipdb;ipdb.set_trace()
                corrs[base_method][((imputation0,order0,train_flag0),(imputation1,order1,train_flag1))] = r.mean()
            
