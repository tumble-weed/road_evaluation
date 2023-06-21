# Implementations of our imputation models.
import torch
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from road.gisp.models.gsmv import GNet_ResNet
import os
import pudb
import colorful
class BaseImputer():
    def __call__(self, img: torch.Tensor, mask: torch.Tensor)-> torch.Tensor:
        """ Call the Imputation function to fill the masked pixels in an image.
            :param img: original image (C,H,W)-tensor
            :param mask: (H,W)-tensor with a binary mask. 0 indicates pixels absent, 1 indicates pixels present.
            :returns: a (C,H,W) tensor, where the original values are kept, if the mask for the pixels is 1 or imputed otherwise.
            The return tensor is copied to cpu()
        """
        raise NotImplementedError("Please implement an imputation function or use an existing imputor.")

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Call the Imputation function to fill the masked pixels in an image. However, in this version,
            and entire batch of images will be processed, which can results in considerable speedup.
            :param img: B original images (B, C, H, W)-tensor
            :param mask: (B, H, W)-tensor with binary masks. 0 indicates pixels absent, 1 indicates pixels present.
            :returns: a (B, C, H, W) tensor, where the original values are kept, if the mask for the pixels is 1 or imputed otherwise.
            The returned tensor is left on the device that this dataloader is instructed to use (may not be CPU).
        """
        raise NotImplementedError("This imputer does not support the bached interface.")

# Some Imputors
class ChannelMeanImputer(BaseImputer):
    """ Impute by the mean value per channel of the image. """

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        for c in range(len(img)):
            mean_c = img[c,:,:].mean()
            imgsubtensor = img[c,:,:]
            imgsubtensor[mask==0] = mean_c
        return img

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor):
        channel_mean_tensor = img.view(img.shape[0], img.shape[1], -1).mean(axis=2) # [B, C]
        c_shape = channel_mean_tensor.shape
        channel_mean_tensor = channel_mean_tensor.unsqueeze(2).unsqueeze(3).expand(c_shape[0], c_shape[1], img.shape[2], img.shape[3])
        return (channel_mean_tensor * (1.0-mask.unsqueeze(1))) + img*mask.unsqueeze(1)

class ZeroImputer(BaseImputer):
    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        return img*mask.unsqueeze(0)
    
    def batched_call(self, img: torch.Tensor, mask: torch.Tensor):
        assert img.device == mask.device
        return img*mask.unsqueeze(1)

class GAINImputer(BaseImputer):
    """
    Imputer based on Generative imputation networks (GAIN).
    See "Generative Imputation and Stochastic Prediction" (2020) by Kachuee et al. for details of the code
    """
    def __init__(self, model_file: str, use_device="cpu"):
        """
            model_file: Path of the pretrained imputation model that should be loaded.
        """
        res_dict = torch.load(model_file, map_location="cpu")
        self.gain_model = GNet_ResNet(n_downsampling=1, n_blocks=4, attention=False)
        self.run_on_device = use_device
        self.mfv = res_dict["mfv"].to(self.run_on_device)
        self.gain_model.load_state_dict(res_dict["generator"])
        self.gain_model.eval()
        self.gain_model.to(use_device)
       

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        img = img.to(self.run_on_device)
        mask = mask.to(self.run_on_device)
        img = (img-self.mfv)
        input_img = torch.where(mask==0, torch.tensor(float("nan"), dtype=torch.float32, device=self.run_on_device), img) # set all other values in the image to zero.
        #print(input_img[:, :10,:10], mask[:10,:10])
        with torch.no_grad():
            x_g =  self.gain_model(input_img.unsqueeze(0)).squeeze(0) # call generator.
        img_imputed = img*mask.unsqueeze(0) + x_g*(1-mask.unsqueeze(0))
        return (img_imputed + self.mfv).cpu()

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor):
        img = img.to(self.run_on_device)
        mask = mask.to(self.run_on_device)
        img = (img-self.mfv.unsqueeze(0))
        input_img = torch.where(mask.unsqueeze(1)==0, torch.tensor(float("nan"), dtype=torch.float32, device=self.run_on_device), img) # set all other values in the image to zero.
        #print(input_img[2, 1, :10,:10], mask[2,:10,:10])
        with torch.no_grad():
            x_g =  self.gain_model(input_img) # call generator.
        img_imputed = img*mask.unsqueeze(1) + x_g*(1-mask.unsqueeze(1))
        return (img_imputed + self.mfv.unsqueeze(0)) # Return tensor is left on device.

# Code for infilling.
neighbors_weights = [((1,1), 1/12), ((0,1), 1/6), ((-1,1), 1/12), ((1,-1), 1/12), ((0,-1), 1/6), ((-1,-1), 1/12), ((1,0), 1/6), ((-1,0), 1/6)]

class NoisyLinearImputer(BaseImputer):
    def __init__(self, noise=0.01, weighting=neighbors_weights):
        """	
            Noisy linear imputation.	
            noise: magnitude of noise to add (absolute, set to 0 for no noise)
            weighting: Weights of the neighboring pixels in the computation. 
            List of tuples of (offset, weight)
        """
        self.noise = noise
        self.weighting = weighting
    
    @staticmethod 
    def add_offset_to_indices(indices, offset, mask_shape):
        """ Add the corresponding offset to the indices. Return new indices plus a valid bit-vector. """
        cord1 = indices % mask_shape[1]
        cord0 = indices // mask_shape[1]
        cord0 += offset[0]
        cord1 += offset[1]
        #print(cord1.shape, indices.shape)
        valid = ((cord0 < 0) | (cord1 < 0) | (cord0 >= mask_shape[0]) | (cord1 >= mask_shape[1]))
        return ~valid, indices+offset[0]*mask_shape[1]+offset[1]

    @staticmethod 
    def setup_sparse_system(mask, img, neighbors_weights):
        """ Vectorized version to set up the equation system.
            mask: (H, W)-tensor of missing pixels.
            Image: (H, W, C)-tensor of all values.
            Return (N,N)-System matrix, (N,C)-Right hand side for each of the C channels.
        """
        maskflt = mask.flatten()
        imgflat = img.reshape((img.shape[0], -1))
        #print(imgflat.shape)
        indices = np.argwhere(maskflt==0).flatten() # Indices that are imputed in the flattened mask
        coords_to_vidx= np.zeros(len(maskflt), dtype=int)
        coords_to_vidx[indices] = np.arange(len(indices)) # lookup_indices =
        #print(coords_to_vidx[:10])
        #coords_to_vidx = {(idx[0].item(), idx[1].item()): i for i, idx in enumerate(indices)} # Coordinates to variable index
        numEquations = len(indices)
        A = lil_matrix((numEquations, numEquations)) # System matrix
        b = np.zeros((numEquations, img.shape[0]))
        sum_neighbors = np.ones(numEquations) # Sum of weights assigned
        #print("My indices:", indices[:10])
        #print("Num indices: ", len(indices))
        for n in neighbors_weights:
            offset, weight = n[0], n[1]
            #print("Using: ", offset, weight)
            # Sum of the neighbors.
            # Take out outliers
            valid, new_coords = NoisyLinearImputer.add_offset_to_indices(indices, offset, mask.shape)
            
            valid_coords = new_coords[valid]
            valid_ids = np.argwhere(valid==1).flatten()
            #print(valid_ids[:10], valid_coords[:10])
            #print("Valid:", valid_ids.shape)
            
            # Add values to the right hand-side
            has_values_coords = valid_coords[maskflt[valid_coords] > 0.5]
            has_values_ids = valid_ids[maskflt[valid_coords] > 0.5]
            #print(has_values_ids[:10], has_values_coords[:10])
            #print("Has Values:", has_values_coords.shape)
            b[has_values_ids, :] -= weight*imgflat[:, has_values_coords].T
            
            # Add weights to the system (left hand side)
            has_no_values = valid_coords[maskflt[valid_coords] < 0.5] # Find coordinates in the system.
            variable_ids = coords_to_vidx[has_no_values]
            has_no_values_ids = valid_ids[maskflt[valid_coords] < 0.5]
            
            #print("Has No Values:", has_no_values.shape)
            A[has_no_values_ids, variable_ids] = weight
            
            # Reduce weight for invalid
            #print(np.argwhere(valid==0).flatten()[:10])
            sum_neighbors[np.argwhere(valid==0).flatten()] = sum_neighbors[np.argwhere(valid==0).flatten()] - weight

        A[np.arange(numEquations),np.arange(numEquations)] = -sum_neighbors  
        return A, b

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        """ Our linear inputation scheme. """
        """
        This is the function to do the linear infilling 
        img: original image (C,H,W)-tensor;
        mask: mask; (H,W)-tensor

        """
        imgflt = img.reshape(img.shape[0], -1)
        maskflt = mask.reshape(-1)
        indices_linear = np.argwhere(maskflt==0).flatten() # Indices that need to be imputed.
        # Set up sparse equation system, solve system.
        A, b = NoisyLinearImputer.setup_sparse_system(mask.numpy(), img.numpy(), neighbors_weights)
        res = torch.tensor(spsolve(csc_matrix(A), b), dtype=torch.float)

        # Fill the values with the solution of the system.
        img_infill = imgflt.clone()
        img_infill[:, indices_linear] = res.t() + self.noise*torch.randn_like(res.t())
            
        return img_infill.reshape_as(img)

    def batched_call(self, img: torch.Tensor, mask: torch.Tensor):
        """ Pseudo implementation of batched interface. """
        res_list = []
        in_device = img.device
        for i in range(len(img)):
            res_list.append(self.__call__(img[i].cpu(), mask[i].cpu()))
        return torch.stack(res_list).to(in_device)

def _from_str(imputer_str):
    """ Return a default imputer from a string. """
    if imputer_str == "linear":
        return NoisyLinearImputer()
    elif imputer_str == "fixed":
        return ChannelMeanImputer()
    elif imputer_str == "zero":
        return ZeroImputer()
    elif (imputer_str == "gan" or imputer_str == "gain"):
        raise ValueError("GAIN imputer cannot be created via default, because a pretrained model " +
                "needs to be passed. Please use the explicit constructor of road.imputations.GAINImputer.")
    else:
        raise ValueError("Unknown imputer string. Please use {linear, fixed, zero}.")
    
    
#==================================================================================
import sys
class AddPath():
    def __init__(self,d):
        print('TODO: move this to another file')
        self.d = d

        pass
    def __enter__(self):
        self.at = len(sys.path)
        sys.path.append(self.d)        
        pass
    def __exit__(self,*args):
        del sys.path[self.at]
        pass


with AddPath('/root/evaluate-saliency-4/GPNN_for_road'):
    import dutils
    from model.my_gpnn_inpainting import gpnn
    # from model.my_gpnn_inpainting import gpnn
    # from model.my_gpnn_inpainting_new import gpnn
    tensor_to_numpy = lambda t:t.detach().cpu().numpy()
    def denormalize_imagenet(t,vgg_mean=[0.485, 0.456, 0.406],
                    vgg_std=[0.229, 0.224, 0.225]):
        device = t.device
        out = (t )*torch.tensor(vgg_std).to(device)[None,:,None,None] + torch.tensor(vgg_mean).to(device)[None,:,None,None]
        return out
    class GPNNImputer:
        def __init__(self):
            print('TODO: how do i add the path to gpnn?')
            print('TODO: eventually will need to move gpnn to this dir')
            """    
                Noisy linear imputation.    
                noise: magnitude of noise to add (absolute, set to 0 for no noise)
                weighting: Weights of the neighboring pixels in the computation. 
                List of tuples of (offset, weight)
            """
        


        def __call__(self, img: torch.Tensor, mask: torch.Tensor):
            if (img < 0).any() or (img > 1).any():
                print('did not expect img to have negative values')
                import ipdb;ipdb.set_trace()
            UPSIZE = True
            img_orig = img
            mask_orig = mask
            if UPSIZE:
                # import ipdb;ipdb.set_trace()
                print(colorful.orchid('move the upsize to inpainting with a flag, and use masked_convolution'))
                UPSIZE_FACTOR = 3
                print(img.shape)
                print(mask.shape)
                up_img = torch.nn.functional.interpolate(img[None,...],size=(img.shape[-2]*UPSIZE_FACTOR,img.shape[-1]*UPSIZE_FACTOR),mode='nearest')
                up_mask = torch.nn.functional.interpolate(mask[None,None,...],size=(img.shape[-2]*UPSIZE_FACTOR,img.shape[-1]*UPSIZE_FACTOR),mode='nearest')
                up_img = up_img[0]
                up_mask = up_mask[0,0]
                img = up_img
                mask = up_mask       
                if 'check' and False:
                    re_img_orig = torch.nn.functional.interpolate(img[None,...],size=img_orig.shape[-2:],mode='nearest')[0]
                    print((img_orig -re_img_orig).abs().sum())
            # import ipdb;ipdb.set_trace()
            with AddPath('/root/evaluate-saliency-4/GPNN_for_road'):
                mask = mask.float()
                config = {
                    'out_dir':'gpnn-eval/output',
                    'iters':1,
                    # 'iters':1,#10
                    'coarse_dim':14,#
                    # 'coarse_dim':28,
                    # 'coarse_dim':100,#
                    'out_size':0,
                    'patch_size':3,
                    # 'patch_size':7,
                    # 'patch_size':15,
                    'stride':1,
                    # 'pyramid_ratio':8/7,
                    'pyramid_ratio':4/3,
                    # 'pyramid_ratio':2,
                    'faiss':True,
                    # 'faiss':False,
                    'no_cuda':False,
                    #---------------------------------------------
                    'in':None,
                    'sigma':4*0.75,
                    # 'sigma':0.3*0.75,
                    'alpha':0.005,
                    'task':'inpainting',
                    #---------------------------------------------
                    # 'input_img':original_imname,
                    # 'input_img':tensor_to_numpy(denormalize_imagenet(img.unsqueeze(0)).permute(0,2,3,1)[0]),
                    'input_img':tensor_to_numpy((img.unsqueeze(0)).permute(0,2,3,1)[0]),
                    # NOTE: the mask arrives with 0's at holes. need to flip it
                    'mask':tensor_to_numpy(1 - mask),
                    'batch_size':10,
                    #---------------------------------------------
                    'implementation':'gpnn',#'efficient-gpnn','gpnn'
                    'init_from':'zeros',#'zeros','target'
                    'keys_type':'single-resolution',#'multi-resolution','single-resolution'
                    #---------------------------------------------
                    'use_pca':False,
                    'n_pca_components':10,
                    #---------------------------------------------
                    'patch_aggregation':'uniform',#'uniform','distance-weighted','median'
                    # 'imagenet_target':imagenet_target
                    #---------------------------------------------
                    'index_type':'brute-force',
                    'use_xy':True,
                    }
                gpnn_inpainting = gpnn(config)
                holefilled,holefilling_results = gpnn_inpainting.run(to_save=False)
                if True:
                    holefilled_bchw = tensor_to_numpy(holefilled[:,:3])
                    mask_bchw = (config['mask'][None,None,...])
                    input_img_bchw = config['input_img'].transpose(2,0,1)[None,...]
                    corruption = np.abs(holefilled_bchw * (1 - mask_bchw) - input_img_bchw* (1 - mask_bchw)).sum()
                    print(corruption)
                                
                # import ipdb;ipdb.set_trace()
                # pudb.set_trace()
                area = config['mask'].sum().item()
                if False:
                    import dutils
                    dutils.img_save(tensor_to_numpy(1-mask),f'imputation_mask_{int(area)}.png')
                    dutils.img_save(tensor_to_numpy(img[:,:3].permute(1,2,0)),f'imputation_img_{int(area)}.png')
                    
                    dutils.img_save(tensor_to_numpy(holefilled[:,:3].permute(0,2,3,1)[0]),f'imputed_{int(area)}.png')



                # """ Our linear inputation scheme. """
                # """
                # This is the function to do the linear infilling 
                # img: original image (C,H,W)-tensor;
                # mask: mask; (H,W)-tensor

                # """
                # imgflt = img.reshape(img.shape[0], -1)
                # maskflt = mask.reshape(-1)
                # # Indices that need to be imputed.
                # indices_linear = np.argwhere(maskflt==0).flatten() 
                # # Set up sparse equation system, solve system.
                # A, b = GPNNImputer.setup_sparse_system(mask.numpy(), 
                #     img.numpy(), neighbors_weights)
                # res = torch.tensor(spsolve(csc_matrix(A), b), dtype=torch.float)

                # # Fill the values with the solution of the system.
                # img_infill = imgflt.clone()
                # img_infill[:, indices_linear] = res.t() + self.noise*torch.randn_like(res.t())
                    
                # return img_infill.reshape_as(img)

                assert holefilled.shape[-2:] == img.shape[-2:]
                assert holefilled.shape[1] in [3,5]
                out_img = holefilled[0,:3]
                out_img = torch.nn.functional.interpolate(out_img[None,...],img_orig.shape[-2:],mode='nearest')[0]
                # import ipdb;ipdb.set_trace()
                return out_img

        def batched_call(self, img: torch.Tensor, mask: torch.Tensor):
            # import ipdb;ipdb.set_trace()
            """ Pseudo implementation of batched interface. """
            res_list = []
            in_device = img.device
            for i in range(len(img)):
                res_list.append(self.__call__(img[i], mask[i]))
            return torch.stack(res_list).to(in_device)    
    #================================================================================
DummyGPNNImputer = ZeroImputer
