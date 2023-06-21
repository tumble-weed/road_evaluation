# IMPORT_NEW_GPNN=1 pudb NoRetrainingMethod_imagenet.py \
#         --data_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
# 		--expl_path=/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data \
# 		--params_file='noretrain_params_imagenet.json' \
# 		--model_path='../../data/cifar_8014.pth'

import register_ipdb
import dutils
import colorful as cf
from datetime import datetime
import os, sys
import torchvision
from utils import append_evaluation_result, get_missing_run_parameters,get_missing_run_parameters_details, update_eval_result, update_eval_result_details, load_expl, arg_parse
import json
import time
import hack_import
## import from road module
import road
from road import run_road
from road.imputations import *
from road.retraining import *
import pudb
import glob
import pickle
from termcolor import colored
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
TODO = None

# pudb.set_trace()
# different seeds
seeds = [2005, 42, 1515, 3333, 420]
vgg_mean = (0.485, 0.456, 0.406)
vgg_std = (0.229, 0.224, 0.225)
def get_vgg_transform(size=224):
    vgg_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size),
            torchvision.transforms.Normalize(mean=vgg_mean,std=vgg_std),
            ]
        )
    return vgg_transform


def load_imagenet_expl(train_file, test_file):
    '''
    has interface in the form of load_expl
    hardcodes some values to load_imagenet_expl_
    '''
    MODELNAME = 'vgg16'
    SPLIT = 'val'
    RESULTS_DIR = '/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/benchmark/vgg16-results'
    IMAGENET_ROOT = '/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet'    
    fileparts = test_file.split('/')
    pklname = fileparts[-1]
    methodname = fileparts[-2]
    # print('see methodname')
    # import ipdb;ipdb.set_trace()
    return load_imagenet_expl_(            
    methodname = methodname,
    modelname = MODELNAME,
    SPLIT = SPLIT,
    imagenet_root = IMAGENET_ROOT,
    RESULTS_DIR= RESULTS_DIR,
    )
def load_imagenet_expl_(            
methodname = 'smoothgrad',
modelname = 'vgg16',
SPLIT = 'val',
imagenet_root = None,
RESULTS_DIR= None,
):
    '''
    knows how to read the pickle files for the methods
    '''
    
    methoddir =  os.path.join(RESULTS_DIR,f'{methodname}-{modelname}')
    im_save_dirs = list(sorted(glob.glob(os.path.join(methoddir,'*/'))))
    has_errors = []
    class test_dataset_type():
        def __getitem__(self,i):
            # for i,im_save_dir in enumerate(im_save_dirs):
            im_save_dir = im_save_dirs[i]
            print(colored(im_save_dir,'green'))
            imroot = os.path.basename(im_save_dir.rstrip(os.path.sep))
            impath = os.path.join(imagenet_root,'images',SPLIT,imroot + '.JPEG')
            #----------------------------------------------------------------
            pklname = glob.glob(os.path.join(im_save_dir,'*.pkl'))
            assert len(pklname) == 1
            pklname = pklname[0]
            try:
                with open(pklname,'rb') as f:
                    loaded = pickle.load(f)
            except EOFError as e:
                has_errors.append(im_save_dir)
                print(colored(f'error in reading {im_save_dir}','yellow'))
            saliency = loaded['saliency']
            label = loaded['target_id']
            prediction = None
            assert saliency.ndim == 4
            assert saliency.shape[:2] == (1,1)
            saliency = tensor_to_numpy(saliency)[0,0]
            saliency = np.tile(saliency[...,None],(1,1,3))
            return saliency
    test_dataset = test_dataset_type()
    DUMMY_TEST_PREDICTION = lambda i:0
    return None,test_dataset,None,DUMMY_TEST_PREDICTION
if __name__ == '__main__':
    ## read configs
    args = arg_parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_path = args.data_path
    expl_path = args.expl_path
    use_device = torch.device("cuda" if args.gpu else "cpu")
    batch_size = args.batch_size
    params_file = args.params_file
    dataset = args.dataset
    model_path = args.model_path
    num_of_classes = 1000
    ## set transforms
    if False:
        transform_train = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #transforms.RandomHorizontalFlip(),
        transform_test = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform_train = get_vgg_transform(size=224)
        transform_test = get_vgg_transform(size=224)


    # Apply this transformation after imputation.
    if False:
        normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        normalize_transform = torchvision.transforms.Normalize(mean=vgg_mean,std=vgg_std)
    
    # import ipdb;ipdb.set_trace()
    params = json.load(open(params_file))
    print("Base Method Group: ", params["basemethod"])
    print("Types:", params["modifiers"])
    print("Imputation: ", params["imputation"])
    print("MoRF-order", bool(params["morf"]))
    print("Resultsfile",  params["datafile"])
    print("Percentages", params["percentages"])
    print("Timeout", int(params["timeoutdays"]))
    if os.environ.get('DBG_HIGH_PERCENT',False) == '1':
        params['percentages'] = [0.9]
        print(colored(f'setting percentages to high {params["percentages"]}','yellow'))
        import time;time.sleep(5)
    if os.environ.get('DBG_LOW_PERCENT',False) == '1':
        params['percentages'] = [0.1,0.2,0.3]
        print(colored(f'setting percentages to low {params["percentages"]}','yellow'))
        import time;time.sleep(5)            
    imputation = params["imputation"]
    group = params["basemethod"]
    morf = bool(params["morf"])
    storage_file = params["datafile"]
    storage_file_details = storage_file[:-len('.json')] + '_details.json'
    modifiers = params["modifiers"]
    ps = params["percentages"]


    target_num_runs = 1
    if imputation == "linear":
        imputer = NoisyLinearImputer(noise=0.01)
        target_num_runs = 5
    elif imputation == "gpnn":
        # imputer = GPNNImputer()        
        imputer = DummyGPNNImputer()
    elif imputation == "fixed":
        imputer = ChannelMeanImputer()
    elif imputation == "gain":
        assert False,'gain not available for imagenet'
        imputer = GAINImputer("../../road/gisp/models/cifar_10_best.pt", "cuda")

    print('imagenet model')

    if 'cifar' and False:
        # Load trained model
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)
        # load trained classifier
        model.load_state_dict(torch.load(model_path,map_location=next(model.parameters()).device))
    else:
        model = models.vgg16(pretrained=True)
        model.to('cuda' if args.gpu else 'cpu')
        model.eval()

    # print('TODO:what is there in run params?')
    # from ipdb import set_trace as set_trace76;set_trace76()
    if True:
        run_params = get_missing_run_parameters(storage_file, imputation, morf, group, modifiers, ps, timeout=int(params["timeoutdays"]),target_num_runs=1)
        run_params_ = get_missing_run_parameters_details(storage_file_details, imputation, morf, group, modifiers, ps, timeout=int(params["timeoutdays"]))
    else:
        print(colored('fix get_missing_run_parameters','yellow'))
        run_params = None,0.2,0
        import time;time.sleep(5)
    
    while run_params is not None:
        print("Got Run Parameters (mod, perc, run_id): ", run_params)
        print(cf.orange('NO-RETRAINING'),cf.yellow('IMAGENET'))
        modifier = run_params[0]
        perc_value = run_params[1]
        run_id = run_params[2]
        torch.manual_seed(seeds[run_id]) # set appropriate seed 

        expl_train = f"{expl_path}/{group}/{modifier}_train.pkl"
        expl_test = f"{expl_path}/{group}/{modifier}_test.pkl"

        start_time = time.time()
        ## load cifar 10 dataset in tensors
        print('TODO:correct for size for vgg16 and resnet50?\nyes VGG16 also takes 244 input ( not necessarily 224x224)')
        # import ipdb;ipdb.set_trace()
        #cifar_train = torchvision.datasets.CIFAR10(root=data_local, train=True, download=True, transform=transform_tensor)
        #============================================================
        transform_tensor_cifar = transforms.Compose([transforms.ToTensor()])
        dataset_test_cifar= torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_tensor_cifar)

        size = 224
        transform_tensor = transforms.Compose([transforms.ToTensor(),
                                            torchvision.transforms.Resize(size)])

        dataset_test2= torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_tensor)
        from imagenet_utils import ImagenetDataset
        dataset_test = ImagenetDataset(transform = transform_tensor)
        #============================================================
        ## load explanation
        #_, explanation_train, _, prediction_train = load_expl(None, expl_train)
        # _, expl_test, _, pred_test = load_expl(None, expl_test)
        import ipdb;ipdb.set_trace()
        # print('load_expl for imagenet')
        # def load_imagenet_expl(*args,**kwargs):
        #     device = 'cuda'
        #     W,H,chan = 298,224,3
        #     expl_test = np.ones((1,H,W,chan))
        #     pred_test = None
        #     return None,expl_test,None,pred_test
        _, expl_test, _, pred_test = load_imagenet_expl(None, expl_test)
        
        
        # pudb.set_trace()
        more_road_returns = {}
        batch_size=1
        res_acc, prob_acc = run_road(model, dataset_test, expl_test, normalize_transform, [perc_value], morf=morf, batch_size=batch_size, imputation = imputer,
        more_returns=more_road_returns)
        # res_acc, prob_acc = run_road(model, dataset_test, expl_test, normalize_transform, [perc_value], morf=morf, batch_size=32, imputation = imputer)

        print('finished job with params', run_params, " Drawing new params.")
        print('--' * 50)
        print("--- %s seconds ---" % (time.time() - start_time))
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # import ipdb;ipdb.set_trace()
        # def update_eval_result_details(!, storage_file, imputation, group, modifier, morf, perc_value, run_id):

        update_eval_result_details(more_road_returns['test_predictions'][perc_value],
            more_road_returns['test_probs'][perc_value], storage_file_details, imputation, group, modifier, morf, perc_value, run_id)
        update_eval_result(res_acc[0].item(), storage_file, imputation, group, modifier, morf, perc_value, run_id)
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        print('TODO: see how run_params change')
        # from ipdb import set_trace as set_trace107;set_trace107()
        run_params = get_missing_run_parameters(storage_file, imputation, morf, group, modifiers, ps,target_num_runs=1)
        print("Got Run Parameters (mod, perc, run_id): ", run_params)
        # exit()

    print("No more open runs. Terminiating.")
