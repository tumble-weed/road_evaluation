import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
# from collections import defaultdict
from termcolor import colored
#sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
from imagenet_localization_parser import get_voc_label
from synset_utils import synset_id_to_imagenet_class_ix
import torchvision
import sys
TODO = None
#=========================================
sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
IGNORE_MULTIPLE_OBJECTS=True
class ImagenetDataset(Dataset):
    #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
    def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val',transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(self.root_dir,'images',split)
        self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
        self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
        self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
        self.IGNORE_MULTIPLE_OBJECTS = IGNORE_MULTIPLE_OBJECTS
        # self.objects = defaultdict(list)
        self.object_synsets = {}
        self.object_to_image = {}
        obj_idx = 0
        for idx in range(len(self.image_paths)):
            bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            for obj in bbox['annotation']['object']:
                # self.objects[idx].append(obj['name'])
                self.object_to_image[obj_idx] = idx
                self.object_synsets[obj_idx] = obj['name']
                obj_idx += 1
                if IGNORE_MULTIPLE_OBJECTS:
                    break
        
        '''
        get_bndbox(
        root_dir = voc_root_dir,
        x = '000003')
        '''
        assert all([file.endswith('.JPEG') for file in self.image_paths])
        assert all([file.endswith('.xml') for file in self.bbox_paths])

    def __len__(self):
        #return len(self.image_paths)
        return len(self.object_to_image)

    def __getitem__(self, idx):
        #from imagenet_localization_parser import get_voc_label
        obj_idx = idx
        im_idx = self.object_to_image[obj_idx]
        image = Image.open(self.image_paths[im_idx])
        print(colored(self.image_paths[im_idx],'magenta'))
        if self.transform is not None:
            image = self.transform(image)
        synset = self.object_synsets[obj_idx]
        class_ix = synset_id_to_imagenet_class_ix(synset)
        if self.IGNORE_MULTIPLE_OBJECTS:
            print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
        #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
        #self.name = bbox['annotation']['object'][0]['name']
        #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
        #print(bbox['annotation']['object'][0]['name'])
        # perform any necessary pre-processing on the image
        # ...
        return image,class_ix
###############################
# VGG transform
###############################
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

if __name__ == '__main__':
    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    img,class_ix = dataset[0]

