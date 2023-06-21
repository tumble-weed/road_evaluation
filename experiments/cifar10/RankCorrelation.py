##
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import scipy
from sklearn.linear_model import LinearRegression
import time
import icecream as ic
import pudb
import numpy as np
from termcolor import colored
import json
import numpy as np
from icecream import ic
##
# some functions
def load_scores():
    superfiles = ['result/retrain_details.json','result/noretrain_details.json']
    
    for j in superfiles:
        with open(j,'r') as f:
            data = json.load(f)
        print(data.keys())
        imputations = data['imputations']
        base_methods = data['base_methods']
        percentages = data['percentages']
        modifiers = data['modifiers']
        orders = data['orders']
        print('is this need to be combinatino or is this a simple concatenation')
        #import time;time.sleep(5)
        if False:
            #attribution_methods = [f'{b}-{m}' for b in base_methods for m in modifiers]
            #scores = np.zeros((len(attribution_methods),NRUNS,len(perc)))
            pass
        attribution_methods = range(8)
        # TODO_NRUNS = 5
        probs = {}
        predictions = {}


        for ii,imputation in enumerate(imputations):
            for ib,base_method in enumerate(base_methods):
                for im,modifier in enumerate(modifiers):
                    for io,order in enumerate(orders):
                        for ip,percentage in enumerate(percentages):
                            print(imputation,base_method,modifier,order)

                            predictions_ = data[imputation][base_method][modifier][order][str(percentage)]['predictions']
                            probs_ = data[imputation][base_method][modifier][order][str(percentage)]['probs']
                            if imputation not in probs:
                                nruns = len(predictions_)
                                nsamples = predictions_[0].__len__()
                                if False:
                                    probs[imputation] = np.zeros((nsamples,len(attribution_methods),nruns,len(percentages)))
                                    predictions[imputation] = np.zeros((nsamples,len(attribution_methods),nruns,len(percentages)))
                                else:
                                    probs[imputation] = np.zeros((10000,8,5,1))
                                    predictions[imputation] = np.zeros((10000,8,5,1))
                            predictions_ = np.array(predictions_)
                            probs_ = np.array(probs_)
                            probs[imputation][:,ib*len(modifiers)+im,:,ip] = probs_.transpose(1,0)
                            predictions[imputation][:,ib*len(modifiers) + im,:,ip] = predictions_.transpose(1,0)
                        

    return probs,predictions
if False:
    def load_scores(roar, file_path='for_review', average=True, run=None):
        '''
            This function load score sperately for roar and kar by setting the flog roar or kar.
            does run let you load only for a specific run?
        '''
        if roar:
            method = "roar"
        else:
            method = "kar"
        scorelist = []
        # 'data/retrain/fixed/ig/roar/base.txt'
        #return np.zeros((10,))
        print(colored('mocking morf_scores_ret_linear','yellow'))
        attribution_methods = ["IG-Base", "IG-SG", "IG-SQ", "IG-Var", "GB-Base", "GB-SG", "GB-SQ", "GB-Var"]
        perc = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
        NRUNS = 5
        scores = np.zeros((len(attribution_methods),NRUNS,len(perc)))
        return scores
        for datafile in ["%s/ig/"%file_path+method+"/base.txt", "%s/ig/"%file_path+method+"/sg.txt", "%s/ig/"%file_path+method+"/sq.txt", \
                        "%s/ig/"%file_path+method+"/var.txt", "%s/gb/"%file_path+method+"/base.txt", "%s/gb/"%file_path+method+"/sg.txt", \
                        "%s/gb/"%file_path+method+"/sq.txt", "%s/gb/"%file_path+method+"/var.txt"]:
            with open(datafile,'r') as f:
                base_list = []
                for m,line in enumerate(f):
                    # base = []
                    if roar:
                        base = [0.80]
                    else:
                        base = [0.30]
                    if run is None:
                        if m <= 4:
                            base.extend([float(i) for i in line.strip('\n').split(' ')[:-1]])
                            base_list.append(base)
                    else:
                        if m == run:
                            base.extend([float(i) for i in line.strip('\n').split(' ')[:-1]])
                            base_list.append(base)
                base_list = np.asarray(base_list)
                scores = np.vstack(base_list)
                # std_em = scipy.stats.sem(scores, axis=0)
                # max_std_em = max(std_em)
                # scorelist.append(max_std_em)
                if average:
                    scores = np.mean(scores,axis=0)
                    scorelist.append(scores)
                # print(ig.shape)
                else:
                    #print(scores.shape)
                    scorelist.append(scores)
            # scorelist.append(ig.reshape(1,-1))
        scores = np.stack(scorelist)
        #scores mght be  [scores(ig);scores(sq);...] 
        #scores of 1 method at a time
        return scores

##
'''
Load the data of the run. Naming: {``morf``,``lerf``}``_``{``ret``,``non``}``_``{``fixed``,``linear``}, where the first gap give the order (MoRF=roar, LeRF=kar), the second gives retrain or non-retrain and the third provides the inpainting strategy used.
'''
##
if False:
    morf_scores_ret_fixed = load_scores(roar=True, file_path='data/retrain/fixed', average=False, run=None)
    morf_scores_ret_linear = load_scores(roar=True, file_path='data/retrain/linear', average=False, run=None)
    morf_scores_non_fixed = load_scores(roar=True, file_path='data/nonretrain/fixed', average=False, run=None)
    morf_scores_non_linear = load_scores(roar=True, file_path='data/nonretrain/linear', average=False, run=None)
    ##
    lerf_scores_ret_fixed = load_scores(roar=False, file_path='data/retrain/fixed', average=False, run=None)
    lerf_scores_ret_linear = load_scores(roar=False, file_path='data/retrain/linear', average=False, run=None)
    lerf_scores_non_fixed = load_scores(roar=False, file_path='data/nonretrain/fixed', average=False, run=None)
    lerf_scores_non_linear = load_scores(roar=False, file_path='data/nonretrain/linear', average=False, run=None)
    print(morf_scores_ret_fixed.shape)
##
scores = load_scores()

def plot_scores(score_matrix, caption):
    import ipdb;ipdb.set_trace()
    colors = ["g", "b", "m", "r", "y", "c", "orange", "lime"]
    methods = ["IG-Base", "IG-SG", "IG-SQ", "IG-Var", "GB-Base", "GB-SG", "GB-SQ", "GB-Var"]
    perc = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    for i in range(len(colors)):
        linestyle = "--" if i >=4 else "-"
        #print(colored('mocking score_matrix','yellow'))
        #import time;time.sleep(1)
        #NRUNS = 5
        #score_matrix = np.zeros((len(methods),NRUNS,len(perc)))
        plt.plot(perc, np.mean(score_matrix[i,:,:], axis=0), color = colors[i], linestyle=linestyle, label=methods[i])
    plt.xlabel("percent removed / inserted")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title(caption)
## Some examples
plot_scores(scores)
if False:
    plot_scores(morf_scores_ret_fixed, "MoRF, retrain, fixed")
    plot_scores(morf_scores_ret_linear, "MoRF, retrain, linear")
    plot_scores(morf_scores_non_linear, "MoRF, non-retrain, linear")
    # While the actual values of the retrain and non-retrain curves start to differ after a few percent removed, the order or the methods stays largely intact.
    plot_scores(lerf_scores_ret_linear, "LeRF, retrain, linear")
    ## Take a first glance at the orders.
    names_list = ["IG-Base", "IG-SG", "IG-SQ", "IG-Var", "GB-Base", "GB-SG", "GB-SQ", "GB-Var"]
    ##
    # Order for 10 percent with retraining and linear
    #print('mocking morf_scores_ret_linear','yellow')
    #methods = ["IG-Base", "IG-SG", "IG-SQ", "IG-Var", "GB-Base", "GB-SG", "GB-SQ", "GB-Var"]
    #perc = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    #morf_scores_ret_linear = np.zeros((len(methods),NRUNS,len(perc)))
    #morf_scores_non_linear = np.zeros((len(methods),NRUNS,len(perc)))
    print([names_list[i] for i in np.argsort(np.mean(morf_scores_ret_linear[:,:,1], axis=1))])
    ##
    # Order for 10 percent
    print([names_list[i] for i in np.argsort(np.mean(morf_scores_non_linear[:,:,1], axis=1))])
# We observe that only GB-SG changes its IG-Var changes position in the ranking significantly, while the first for order stay mostly similar.
# We now want to analyze this in a more quantitative and principled way and compute the rank correlation of the approaches. We use the Spearman Rank Correlation to this end.
# To compute an approximate measure of uncertainty, we use a bagging approach and sample data from three random runs. We do so for 10 times and compute the std deviation of the rank-coefficients.
if False:
    
    allres = np.stack([morf_scores_ret_fixed, morf_scores_ret_linear, morf_scores_non_linear, morf_scores_non_fixed,
                    lerf_scores_ret_fixed, lerf_scores_ret_linear, lerf_scores_non_linear, lerf_scores_non_fixed])
    #allres shape will be  (n_imputations,) + (n_methods,n_runs,n_percentages) 
    print(allres.shape)
else:
    import ipdb;ipdb.set_trace()

def compute_corr(ranks):
    """ Compute spearman rank correlation of the rank matrix. 
        ranks = (M, R) matrix (M-methods, R-Ranks in range [1...R])
        return (M,M) matrix with rank correlations between methods.
    """
    #TODO: is MxR: imputation times number of attribution methods?
    # returns correlation between imputation methods
    num_input = ranks.shape[0]
    #num_input: num_imputation_methods
    resmat = np.zeros((num_input,num_input))
    for i in range(num_input):
        for j in range(num_input):
            cov = np.mean(ranks[i]*ranks[j], axis=0)-np.mean(ranks[i], axis = 0)*np.mean(ranks[j], axis=0) # E[XY]-E[Y]E[X]
            corr = cov/(np.std(ranks[i], axis=0)*np.std(ranks[j], axis=0))
            #print(corr)
            resmat[i,j] = np.mean(corr)
    return resmat

def rank_corr_avg(allres):
    """ Return the rank correlation of the average curves. 
        (This leads to results in the paper)
    """
    allres_measured= np.mean(allres[:,:,:,1:], axis=2)
    # shape: n_impute,n_methods,n_perc - 1
    order = allres_measured.argsort(axis=1)
    ranks = order.argsort(axis=1)
    # get the ranks of the methods
    return compute_corr(ranks)
                          
def rank_corr_wstd(allres, bagging=False, lerf_id=[3,4,5,6]):
    """ Return rank correlation matrix by run and compute std. dev matrix 
        (This leads to the results and std deviations shown in the supplement)
        rank corr with std
    """
    num_bagging_runs = 10
    use_runs = 3
    num_runs = allres.shape[2]
    rank_corrs = []
    useruns = num_bagging_runs if bagging else num_runs
    for i in range(useruns):
        if bagging:
            selected_runs = np.random.permutation(num_runs)[:use_runs]
            print(selected_runs)
            allres_measured = np.mean(allres[:,:,selected_runs,1:], axis=2)
        else:
            allres_measured= allres[:,:,i,1:]
        #allres_measured shape: (n_imputations,n_methods,n_perc)
        order = allres_measured.argsort(axis=1)
        #order sorted by methodname (n_imputations,n_methods,n_perc)
        ranks = order.argsort(axis=1)
        if i in lerf_id:
            ranks = ranks[:,::-1]
        rank_corrs.append(compute_corr(ranks))
    rank_corrs = np.stack(rank_corrs)
    print(rank_corrs.shape)
    return rank_corrs.mean(axis=0), rank_corrs.std(axis=0)/np.sqrt(useruns)
##
## Results of correlations by runs (shown in supplementary)
# Print the correlation matrix from the mean of runs. Note that for the MoRF and LeRF cross combinations the sign is inverted, because good ranks indicate good performance in LeRF but bad performance in MoRF.
if False:
    import pandas as pd
    # lin,fix ( linear and fixed)
    # morf and lerf
    # (non)-retrain and (re)train
    index_names_short = ["m-re-fix", "m-re-lin", "m-non-lin", "m-non-fix", "l-re-fix", "l-re-lin", "l-non-lin", "l-non-fix"]
    index_names_full = ["morf-re-fix", "morf-re-lin", "morf-non-lin", "morf-non-fix", "lerf-re-fix", "lerf-re-lin", "lerf-non-lin", "lerf-non-fix"]
    np.set_printoptions(precision=3)
    mean_corr, std_corr = rank_corr_wstd(allres[:,:,])
    panda_df = pd.DataFrame(data = mean_corr, 
                            index = index_names_full, 
                            columns = index_names_short)
    #pd.set_option("precision", 2)
    panda_df
    ##
    # Matrix of std deviations (x 10^{-2})
    panda_df = pd.DataFrame(data = std_corr*100, 
                            index = index_names_full, 
                            columns = index_names_short)
    panda_df
else:
    import ipdb;ipdb.set_trace()

