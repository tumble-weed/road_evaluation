if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os

    class ImageDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.image_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.jpg')]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImageDataset(imagenet_root)
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os

    class ImageDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.image_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.jpg')]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImageDataset(imagenet_root)
    dataset.image_paths[:10]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os

    class ImageDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.image_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.jpg')]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImageDataset(imagenet_root)
    print(dataset.image_paths[:10])
    
os.listdir(imagenet_root)[:2]
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os

    class ImagenetDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images')
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir) if file.endswith('.jpg')]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os

    class ImagenetDataset(Dataset):
        def __init__(self, root_dir):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images')
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os

    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os

    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            assert all([file.endswith('.JPEG') for file in self.image_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os

    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            assert all([file.endswith('.JPEG') for file in self.image_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import imagenet_localization_parser
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    import imagenet_localization_parser
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            assert all([file.endswith('.JPEG') for file in self.image_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    import imagenet_localization_parser
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bboxes = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    
dataset.bboxes[:2]
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    import imagenet_localization_parser
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    import imagenet_localization_parser
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            bbox = imagenet_localization_parser.get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox.keys())
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    import imagenet_localization_parser
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            bbox = imagenet_localization_parser.get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox.keys())
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
import imagenet_localization_parser
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    import imagenet_localization_parser
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            bbox = imagenet_localization_parser.get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox.keys())
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox.keys())
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx])
            bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox.keys())
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            from imagenet_localization_parser import get_voc_label
            image = Image.open(self.image_paths[idx])
            bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox.keys())
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            from imagenet_localization_parser import get_voc_label
            image = Image.open(self.image_paths[idx])
            bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox['annotation'].keys())
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            from imagenet_localization_parser import get_voc_label
            image = Image.open(self.image_paths[idx])
            bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox['annotation']['object'].keys())
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            from imagenet_localization_parser import get_voc_label
            image = Image.open(self.image_paths[idx])
            bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox['annotation']['object'][0].keys())
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            from imagenet_localization_parser import get_voc_label
            image = Image.open(self.image_paths[idx])
            bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox['annotation']['object'][0]['name'].keys())
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            from imagenet_localization_parser import get_voc_label
            image = Image.open(self.image_paths[idx])
            bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            from imagenet_localization_parser import get_voc_label
            image = Image.open(self.image_paths[idx])
            bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            self.name = bbox['annotation']['object'][0]['name']
            assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
get_ipython().run_line_magic('debug', '')
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    from collections import defaultdict
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            from imagenet_localization_parser import get_voc_label
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
            self.objects = defaultdict(list)
            self.object_synsets = {}
            self.inverse_index = {}
            obj_idx = 0
            for idx in range(len(self.image_paths)):
                bbox = get_voc_label(full_filename = self.bbox_paths[idx])
                for obj in bbox['annotation']['object']:
                    self.objects[idx].append(obj['name'])
                    self.inverse_index[obj_idx] = idx
                    self.object_synsets[obj_idx] = obj['name']
                    obj_idx += 1
            
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            #return len(self.image_paths)
            return len(self.inverse_index)

        def __getitem__(self, idx):
            #from imagenet_localization_parser import get_voc_label
            from benchmark.synset_utils import get_synset_id
            obj_idx = idx
            im_idx = self.inverse_index[obj_idx]
            image = Image.open(self.image_paths[im_idx])
            synset = self.object_synsets[obj_idx]
            class_ix = get_synset_id(synset)
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    from collections import defaultdict
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            from imagenet_localization_parser import get_voc_label
            from collections import defaultdict
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
            self.objects = defaultdict(list)
            self.object_synsets = {}
            self.inverse_index = {}
            obj_idx = 0
            for idx in range(len(self.image_paths)):
                bbox = get_voc_label(full_filename = self.bbox_paths[idx])
                for obj in bbox['annotation']['object']:
                    self.objects[idx].append(obj['name'])
                    self.inverse_index[obj_idx] = idx
                    self.object_synsets[obj_idx] = obj['name']
                    obj_idx += 1
            
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            #return len(self.image_paths)
            return len(self.inverse_index)

        def __getitem__(self, idx):
            #from imagenet_localization_parser import get_voc_label
            from benchmark.synset_utils import get_synset_id
            obj_idx = idx
            im_idx = self.inverse_index[obj_idx]
            image = Image.open(self.image_paths[im_idx])
            synset = self.object_synsets[obj_idx]
            class_ix = get_synset_id(synset)
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    _ = dataset[0]
    
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    from collections import defaultdict
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            from imagenet_localization_parser import get_voc_label
            from collections import defaultdict
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
            self.objects = defaultdict(list)
            self.object_synsets = {}
            self.inverse_index = {}
            obj_idx = 0
            for idx in range(len(self.image_paths)):
                bbox = get_voc_label(full_filename = self.bbox_paths[idx])
                for obj in bbox['annotation']['object']:
                    self.objects[idx].append(obj['name'])
                    self.inverse_index[obj_idx] = idx
                    self.object_synsets[obj_idx] = obj['name']
                    obj_idx += 1
            
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            #return len(self.image_paths)
            return len(self.inverse_index)

        def __getitem__(self, idx):
            #from imagenet_localization_parser import get_voc_label
            from benchmark.synset_utils import get_synset_id
            obj_idx = idx
            im_idx = self.inverse_index[obj_idx]
            image = Image.open(self.image_paths[im_idx])
            synset = self.object_synsets[obj_idx]
            class_ix = get_synset_id(synset)
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    img,class_ix = dataset[0]
    
class_ix
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    from collections import defaultdict
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            from imagenet_localization_parser import get_voc_label
            from collections import defaultdict
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
            self.objects = defaultdict(list)
            self.object_synsets = {}
            self.inverse_index = {}
            obj_idx = 0
            for idx in range(len(self.image_paths)):
                bbox = get_voc_label(full_filename = self.bbox_paths[idx])
                for obj in bbox['annotation']['object']:
                    self.objects[idx].append(obj['name'])
                    self.inverse_index[obj_idx] = idx
                    self.object_synsets[obj_idx] = obj['name']
                    obj_idx += 1
            
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            #return len(self.image_paths)
            return len(self.inverse_index)

        def __getitem__(self, idx):
            #from imagenet_localization_parser import get_voc_label
            from benchmark.synset_utils import synset_id_to_imagenet_class_ix
            obj_idx = idx
            im_idx = self.inverse_index[obj_idx]
            image = Image.open(self.image_paths[im_idx])
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    img,class_ix = dataset[0]
    
class_ix
get_ipython().run_line_magic('logstart', 'ipython_imagenet_for_road.py append')
if True:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import os
    import sys
    from collections import defaultdict
    #sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    from imagenet_localization_parser import get_voc_label
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir,split='val'):
            from imagenet_localization_parser import get_voc_label
            from collections import defaultdict
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
            self.objects = defaultdict(list)
            self.object_synsets = {}
            self.inverse_index = {}
            obj_idx = 0
            for idx in range(len(self.image_paths)):
                bbox = get_voc_label(full_filename = self.bbox_paths[idx])
                for obj in bbox['annotation']['object']:
                    self.objects[idx].append(obj['name'])
                    self.inverse_index[obj_idx] = idx
                    self.object_synsets[obj_idx] = obj['name']
                    obj_idx += 1
            
            '''
            get_bndbox(
            root_dir = voc_root_dir,
            x = '000003')
            '''
            assert all([file.endswith('.JPEG') for file in self.image_paths])
            assert all([file.endswith('.xml') for file in self.bbox_paths])

        def __len__(self):
            #return len(self.image_paths)
            return len(self.inverse_index)

        def __getitem__(self, idx):
            #from imagenet_localization_parser import get_voc_label
            from benchmark.synset_utils import synset_id_to_imagenet_class_ix
            obj_idx = idx
            im_idx = self.inverse_index[obj_idx]
            image = Image.open(self.image_paths[im_idx])
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix

    # create an instance of the dataset
    imagenet_root = '/root/evaluate-saliency-4/jigsaw/imagenet'
    dataset = ImagenetDataset(imagenet_root)
    print(dataset.image_paths[:10])
    img,class_ix = dataset[0]
    
if True:
    v = dataset_test[0]
    
if True:
    dataset_test[0].__len__()
    
if True:
    ic(dataset_test[0])
    
from icecream import ic
if True:
    ic(dataset_test[0])
    
if True:
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    
if True:
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
            
if True:
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
            
get_ipython().run_line_magic('logstart', 'ipython_imagenet_for_road.py append')
if True:
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
            
if True:
    from torch.utils.data import Dataset
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
            
if True:
    from torch.utils.data import Dataset
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
            
if True:
    from torch.utils.data import Dataset
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    class ImagenetDataset(Dataset):
        def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
            
if True:
    from torch.utils.data import Dataset
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
            
if True:
    from torch.utils.data import Dataset
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
            
if True:
    from torch.utils.data import Dataset
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
    dataset_test2 = IagenetDataset()
    
if True:
    from torch.utils.data import Dataset
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
    dataset_test2 = ImagenetDataset()
    
if True:
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
    dataset_test2 = ImagenetDataset()
    
if True:
    #=========================================
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    #=========================================
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
    dataset_test2 = ImagenetDataset()
    
if True:
    #=========================================
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    #=========================================
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            from imagenet_localization_parser import get_voc_label
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
    dataset_test2 = ImagenetDataset()
    
if True:
    #=========================================
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    #=========================================
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            from imagenet_localization_parser import get_voc_label
            IGNORE_MULTIPLE_OBJECTS=True
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
    dataset_test2 = ImagenetDataset()
    
if True:
    #=========================================
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    #=========================================
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            from imagenet_localization_parser import get_voc_label
            IGNORE_MULTIPLE_OBJECTS=True
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
    dataset_test2 = ImagenetDataset()
    dataset_test2[0]
    
if True:
    #=========================================
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    #=========================================
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            from imagenet_localization_parser import get_voc_label
            IGNORE_MULTIPLE_OBJECTS=True
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            self.root_dir = root_dir
            self.image_dir = os.path.join(self.root_dir,'images',split)
            self.annotations_dir = os.path.join(self.root_dir,'bboxes',split)
            self.image_paths = [os.path.join(self.image_dir, file) for file in os.listdir(self.image_dir)]
            self.bbox_paths = [os.path.join(self.annotations_dir, file) for file in os.listdir(self.annotations_dir)]
            
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
            from benchmark.synset_utils import synset_id_to_imagenet_class_ix
            obj_idx = idx
            im_idx = self.object_to_image[obj_idx]
            image = Image.open(self.image_paths[im_idx])
            synset = self.object_synsets[obj_idx]
            class_ix = synset_id_to_imagenet_class_ix(synset)
            if IGNORE_MULTIPLE_OBJECTS:
                print(colored('IGNORE_MULTIPLE_OBJECTS (remove the hack before posting)','yellow'))
            #bbox = get_voc_label(full_filename = self.bbox_paths[idx])
            #self.name = bbox['annotation']['object'][0]['name']
            #assert len(bbox['annotation']['object']) == 1,'cannot handle more than 1 object'
            #print(bbox['annotation']['object'][0]['name'])
            # perform any necessary pre-processing on the image
            # ...
            return image,class_ix
    dataset_test2 = ImagenetDataset()
    dataset_test2[0]
    
if True:
    #=========================================
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    #=========================================
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val'):
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            from imagenet_localization_parser import get_voc_label
            IGNORE_MULTIPLE_OBJECTS=True
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            self.root_dir = root_dir
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
            from benchmark.synset_utils import synset_id_to_imagenet_class_ix
            obj_idx = idx
            im_idx = self.object_to_image[obj_idx]
            image = Image.open(self.image_paths[im_idx])
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
    dataset_test2 = ImagenetDataset()
    dataset_test2[0]
    
ic(dataset_test2[0])
transform_tenso
transform_tensor
if True:
    #=========================================
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    #=========================================
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val',transform=None):
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            from imagenet_localization_parser import get_voc_label
            IGNORE_MULTIPLE_OBJECTS=True
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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
            from benchmark.synset_utils import synset_id_to_imagenet_class_ix
            obj_idx = idx
            im_idx = self.object_to_image[obj_idx]
            image = Image.open(self.image_paths[im_idx])
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
    dataset_test2 = ImagenetDataset(transform=None)
    dataset_test2[0]
    
if True:
    #=========================================
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    #=========================================
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val',transform=None):
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            from imagenet_localization_parser import get_voc_label
            IGNORE_MULTIPLE_OBJECTS=True
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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
            from benchmark.synset_utils import synset_id_to_imagenet_class_ix
            obj_idx = idx
            im_idx = self.object_to_image[obj_idx]
            image = Image.open(self.image_paths[im_idx])
            if self.transform is not None:
                image = self.transform(image)
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
    dataset_test2 = ImagenetDataset(transform=None)
    dataset_test2[0]
    
if True:
    #=========================================
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    #=========================================
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val',transform=None):
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            from imagenet_localization_parser import get_voc_label
            IGNORE_MULTIPLE_OBJECTS=True
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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
            from benchmark.synset_utils import synset_id_to_imagenet_class_ix
            obj_idx = idx
            im_idx = self.object_to_image[obj_idx]
            image = Image.open(self.image_paths[im_idx])
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
    dataset_test2 = ImagenetDataset(transform=None)
    dataset_test2[0]
    
ic(dataset_test2[0])
dataset_test[0].min()
dataset_test[0][0].min()
dataset_test[0][0].max()
dataset_test[2][0].max()
dataset_test[2][0].min()
map(el:el[0].min(),dataset_test)
map(lambda el:el[0].min(),[el for iel,el in enumerate(dataset_test) if iel < 10])
list(map(lambda el:el[0].min(),[el for iel,el in enumerate(dataset_test) if iel < 10]))
if True:
    #=========================================
    import sys
    sys.path.append('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN')
    #=========================================
    from torch.utils.data import Dataset
    from imagenet_localization_parser import get_voc_label
    from benchmark.synset_utils import synset_id_to_imagenet_class_ix
    img_tensor,cix = dataset_test[0]
    print(img_tensor.shape)
    IMAGENET_ROOT_DIR="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet"
    IGNORE_MULTIPLE_OBJECTS=True
    class ImagenetDataset(Dataset):
        #def __init__(self, root_dir=IMAGENET_ROOT_DIR,split='val'):
        def __init__(self, root_dir="/root/evaluate-saliency-4/evaluate-saliency-4/jigsaw/imagenet",split='val',transform=None):
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            from imagenet_localization_parser import get_voc_label
            IGNORE_MULTIPLE_OBJECTS=True
            #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
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
            from benchmark.synset_utils import synset_id_to_imagenet_class_ix
            obj_idx = idx
            im_idx = self.object_to_image[obj_idx]
            image = Image.open(self.image_paths[im_idx])
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
    dataset_test2 = ImagenetDataset(transform=None)
    print(colored('the images dont have negative values','yellow'))
    dataset_test2[0]
    
if True:
    def load_imagenet_expl(_,_):
        pass
    load_imagenet_expl(None,expl_test)
if True:
    def load_imagenet_expl(_,__):
        pass
    load_imagenet_expl(None,expl_test)
    
f"{expl_path}/{group}/{modifier}_test.pkl"
get_ipython().run_line_magic('logstart', 'ipython_imagenet_for_road.py append')
def load_imagenet_expl(train_file, test_file):
    methodname = 'smoothgrad'
    modelname = 'vgg16'
    results_root_dir = '/root/evaluate-saliency-4/evaluate-saliency-4/bigfiles/results'
    pass
    
def load_imagenet_expl(train_file, test_file):
    methodname = 'smoothgrad'
    modelname = 'vgg16'
    results_root_dir = '/root/evaluate-saliency-4/evaluate-saliency-4/bigfiles/results'
    
    expl_train = None 
    prediction_train = None
    return expl_train, expl_test, prediction_train, prediction_test
    
def load_imagenet_expl(train_file, test_file):
    methodname = 'smoothgrad'
    modelname = 'vgg16'
    results_root_dir = '/root/evaluate-saliency-4/evaluate-saliency-4/bigfiles/results'
    
    expl_train = None 
    prediction_train = None
    return expl_train, expl_test, prediction_train, prediction_test
load_imagenet_expl(None,None)
def load_imagenet_expl(train_file, test_file):
    methodname = 'smoothgrad'
    modelname = 'vgg16'
    results_root_dir = '/root/evaluate-saliency-4/evaluate-saliency-4/bigfiles/results'
    
    expl_train = None 
    prediction_train = None
    return expl_train, expl_test, prediction_train, prediction_test
load_imagenet_expl(None,None)
expl_path,group,modifier
root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data/ig/base()
get_ipython().system('ls /root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data/ig/base')
get_ipython().system('ls /root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data/ig/base_test.pkl')
with open('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data/ig/base_test.pkl','rb') as f:
    loaded = pickle.load(f)
    
import pickle
with open('/root/evaluate-saliency-4/evaluate-saliency-4/GPNN/road_evaluation/data/ig/base_test.pkl','rb') as f:
    loaded = pickle.load(f)
    
loaded.keys()
ic(loaded)
from icecream import ic
ic(loaded)
ic(loaded[0])
ic(loaded[0]).keys()
ic(loaded[0].keys())
get_ipython().run_line_magic('pinfo', "loaded[0]['expl']")
get_ipython().run_line_magic('whos', "loaded[0]['expl']")
loaded[0]['expl']
ic(loaded[0].keys())
loaded[0]['expl'].__class__
dutils.array_info(loaded[0]['expl'])
def load_imagenet_expl(train_file, test_file):
    methodname = 'smoothgrad'
    modelname = 'vgg16'
    results_root_dir = '/root/evaluate-saliency-4/evaluate-saliency-4/bigfiles/results'
    loaded = {
    'expl':np.zeros((224,224,3))
    }
    expl_train = None 
    prediction_train = None
    return expl_train, expl_test, prediction_train, prediction_test
load_imagenet_expl(None,None)
ic(loaded[0].keys())
ic(loaded[0]['prediction'])
def load_imagenet_expl(train_file, test_file):
    methodname = 'smoothgrad'
    modelname = 'vgg16'
    results_root_dir = '/root/evaluate-saliency-4/evaluate-saliency-4/bigfiles/results'
    loaded = {
    'expl':np.zeros((224,224,3)),
    'prediction':np.array([3]),
    }
    expl_train = None 
    prediction_train = None
    return expl_train, expl_test, prediction_train, prediction_test
load_imagenet_expl(None,None)
ic(loaded[0].keys())
ic(loaded[0]['label'])
ic(loaded[0]['predict_p'])
ic(loaded[0]['predict_p'].shape)
def load_imagenet_expl(train_file, test_file):
    methodname = 'smoothgrad'
    modelname = 'vgg16'
    results_root_dir = '/root/evaluate-saliency-4/evaluate-saliency-4/bigfiles/results'
    loaded = {
    'expl':np.zeros((224,224,3)),
    'prediction':np.array([3]),
    'predict_p':np.zeros((10,)),
    'label':np.array([3]),
    }
    expl_train = None 
    prediction_train = None
    return expl_train, expl_test, prediction_train, prediction_test
load_imagenet_expl(None,None)
ic(loaded[0]['prediction'].shape)
ic(loaded[0]['prediction'])
def load_imagenet_expl(train_file, test_file):
    methodname = 'smoothgrad'
    modelname = 'vgg16'
    results_root_dir = '/root/evaluate-saliency-4/evaluate-saliency-4/bigfiles/results'
    loaded = {
    'expl':np.zeros((224,224,3)),
    'prediction':np.array(3),
    'predict_p':np.zeros((10,)),
    'label':np.array(3),
    }
    expl_train = None 
    prediction_train = None
    return expl_train, expl_test, prediction_train, prediction_test
load_imagenet_expl(None,None)
ic(loaded[0]['label'])
ic(loaded[0]['label'].shape)
def load_imagenet_expl(train_file, test_file):
    methodname = 'smoothgrad'
    modelname = 'vgg16'
    results_root_dir = '/root/evaluate-saliency-4/evaluate-saliency-4/bigfiles/results'
    loaded = {
    'expl':np.zeros((224,224,3)),
    'prediction':np.array(3),
    'predict_p':np.zeros((10,)),
    'label':3,
    }
    expl_train = None 
    prediction_train = None
    return expl_train, expl_test, prediction_train, prediction_test
load_imagenet_expl(None,None)
test_file
'a/b/c/d/e.pkl'.aplit('.')
'a/b/c/d/e.pkl'.split('.')
'a/b/c/d/e.pkl'.split('/')
