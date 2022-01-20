
import random, sys, os, cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from loaders.utils_augmentation import HistoNormalize 



# Credit: some code borrowed and modified from MIL-Nature-Medicine-2019 paper:

class GridWSIPatchDataset(Dataset):
    """
    Data producer that generate all the square grids, e.g. 3x3, of patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """

    def __init__(self,
                 wsi_path=None,
                 mask_path=None,
                 image_size=256,
                 patch_size=256,
                 crop_size=256,
                 normalize=True,
                 full=False,
                 transform=None,
                 level=0,
                 stride=0.5):
        """
        Initialize the data producer.
        Arguments:
            wsi_path: string, path to WSI file
            mask_path: string, path to mask file in numpy format
            image_size: int, size of the image before splitting into grid, e.g.
                768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
            flip: string, 'NONE' or 'FLIP_LEFT_RIGHT' indicating the flip type
            rotate: string, 'NONE' or 'ROTATE_90' or 'ROTATE_180' or
                'ROTATE_270', indicating the rotate type
        """
        self._wsi_path    = wsi_path
        self._mask_path   = mask_path
        self._image_size  = image_size
        self._patch_size  = patch_size
        self._crop_size   = crop_size
        self._normalize   = normalize
        self._full        = full 
        
        self.transform    = transform
        self.slide_level  = level
        self.n_classes    = 2
        self.patch_norm   = HistoNormalize()
        self.stride       = stride
        self._preprocess()

    def _preprocess(self):

        self._mask  = self._mask_path
        self._slide = self._wsi_path

        X_slide, Y_slide = self._slide.level_dimensions[0]
        X_mask, Y_mask   = self._mask.shape

        if self._full:
            # optionally, do not resize the mask, takes longer but more accurate.
            # downsample of 4 corresponds to resolution of (256,256)
            self._mask = cv2.resize(self._mask.copy(), (Y_mask//4, X_mask//4), interpolation=cv2.INTER_NEAREST)
        else:
            # fastest
            self._mask = cv2.resize(self._mask.copy(), (self._patch_size, self._patch_size), interpolation=cv2.INTER_NEAREST)

        X_mask, Y_mask = self._mask.shape
        self._resolution_x = int(X_slide // X_mask)
        self._resolution_y = int(Y_slide // Y_mask)
        print(f'--- Slide {X_slide}, {Y_slide}, {X_mask}, {Y_mask} | res: [{self._resolution_x} |{self._resolution_y}] | Mask: {self._mask.shape} | Full: {self._full}')
        ##

        # all the idces for tissue region from the tissue mask
        # try to resize the mask here
        self._X_idcs, self._Y_idcs = np.where(self._mask)
        self._idcs_num = len(self._X_idcs)


    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]

        # this must be the stride (0.5), increase to have no overlap (1.0)
        x_center = int((x_mask + self.stride) * self._resolution_x)
        y_center = int((y_mask + self.stride) * self._resolution_y)

        m = int(2 ** self.slide_level)
        x = int(x_center - (self._image_size * m) / 2)
        y = int(y_center - (self._image_size * m) / 2)

        img = self._slide.read_region(
            (x, y), self.slide_level, (self._image_size, self._image_size)).convert('RGB')
        
        img = self.patch_norm(img)
        img = img.resize((256,256))
        img = self.transform(img)

        return (img, x_mask, y_mask)


# using folders
class MILFolder(Dataset):

    def __init__(self,
                 root=None,
                 split='val',
                 transform=None,
                 class_map={'normal': 0, 'msi': 1},
                 nslides=-1,
                 train=True):

        self.classmap = class_map
        self.nslides  = nslides
        self.split    = split
        self.root     = root
        self.train    = train
        lib = os.path.join(root,split+'_lib.pth')

        if not os.path.exists(lib):
            """ Format
               root/val/ ...slide1/ patches ...
               root/train/ ... slide_x/ patches ...

            """
            print('Preprocessing folders .... ')
            lib = self.preprocess()
        elif os.path.isfile(lib): 
            print('Using pre-processed lib with tumor patches (>0.75)')
            lib = torch.load(lib)
            
        else: raise ('Please provide root folder or library file')

        self.slidenames = lib['slides']
        self.slides     = lib['slides']
        self.grid       = []
        self.slideIDX   = []
        self.slideLBL   = []
        self.targets    = lib['targets']

        for idx, (slide, g) in enumerate(zip(lib['slides'], lib['grid'])):
            sys.stdout.write('Opening Slides : [{}/{}]\r'.format(idx + 1, len(lib['slides'])))
            sys.stdout.flush()
            self.grid.extend(g)
            self.slideIDX.extend([idx] * len(g))
            self.slideLBL.extend([self.targets[idx]] * len(g))
        print('')
        print(np.unique(self.slideLBL), len(self.slideLBL), len(self.grid))
        print('Number of tiles: {}'.format(len(self.grid)))

        self.transform = transform

    def __getitem__(self, index):
        
        slideIDX = self.slideIDX[index]
        target   = self.targets[slideIDX]
        img      = Image.open(os.path.join(self.slides[slideIDX],self.grid[index])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.grid)
    
    def preprocess(self):
        """
            Change format of lib file to:
            {
                'slides': [xx.tif,xx2.tif , ....],
                'grid'  : [[(x,y),(x,y),..], [(x,y),(x,y),..] , ....],
                'targets': [0,1,0,1,0,1,0, etc]
            }
            len(slides) == len(grid) == len(targets)
        """
        grid    = []
        targets = []
        slides  = []
        class_names = [str(x) for x in range(len(self.classmap))]
        for i, cls_id in enumerate(class_names):
            slide_dicts = os.listdir(os.path.join(self.root, self.split, cls_id))
            #if len(slide_dicts) == 0: continue

            print('--> | ', cls_id, ' | ',self.split, ' | ', len(slide_dicts))

            for idx, slide in enumerate(slide_dicts[:self.nslides]):
                slide_folder = os.path.join(self.root, self.split, cls_id, slide)
                grid_number  = len(os.listdir(slide_folder))
                # skip empty folder
                if grid_number == 0:
                    print("Skipped : ", slide, cls_id,' | ' ,grid_number)
                    continue
                
                if self.train:
                    if grid_number < 32:
                        print("Skipped : ", slide, cls_id, ' | ' ,grid_number)
                        continue

                grid_p = []
                for id_patch in os.listdir(slide_folder):
                    grid_p.append(id_patch)

                slides.append(slide_folder)
                #grid.append(grid_p[:32])
                grid.append(grid_p)
                targets.append(int(cls_id))

        print(len(slides), len(grid), len(targets))
        return {'slides': slides, 'grid': grid, 'targets': targets}


class MILRNNFolder(Dataset):

    def __init__(self, root=None,
                 transform=None,
                 s=10,
                 split='val',
                 shuffle=False,
                 class_map={'normal': 0, 'msi': 1},
                 nslides=-1,
                 train=True):

        self.classmap = class_map
        self.nslides = nslides
        self.split   = split
        self.root    = root
        self.train   = train
        lib = os.path.join(root,split+'_lib.pth')

        if not os.path.exists(lib):
            """ Format
               root/val/ ...slide1/ patches ...
               root/train/ ... slide_x/ patches ...

            """
            print('Preprocessing folders .... ')
            lib = self.preprocess()
        elif os.path.isfile(lib): 
            print('Using pre-processed lib with tumor patches (>0.75)')
            lib = torch.load(lib)
            lib = self.preprocess_lib(lib)
            
        else: raise ('Please provide root folder or library file')

        self.slidenames = lib['slides']
        self.slides     = lib['slides']
        self.grid       = lib['grid']
        self.targets    = lib['targets']

        for idx, (slide, g) in enumerate(zip(lib['slides'], lib['grid'])):
            sys.stdout.write('Opening Slides : [{}/{}]\r'.format(idx + 1, len(lib['slides'])))
            sys.stdout.flush()
        print('')

        self.transform = transform
        self.shuffle   = shuffle
        self.s         = s

    def __getitem__(self, index):
        slide  = self.slides[index]
        grid   = self.grid[index]
        target = self.targets[index]

        if self.shuffle:
           grid = random.sample(grid, len(grid))

        s   = min(self.s, len(grid))
        out = torch.zeros((s, 3, 256, 256))
        out_labels = torch.zeros((s),)

        for i in range(s):
            img = Image.open(os.path.join(slide, grid[i])).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            out[i]        = img
            out_labels[i] = target

        return out, target, out_labels

    def __len__(self):
        return len(self.targets)

    def preprocess_lib(self, lib):

        slides = []
        grid   = []
        targets = []

        for i,  (slide_id, g, t) in enumerate(zip(lib['slides'],lib['grid'],lib['targets'])):
            grid_number  = len(g)
            slide_name = os.path.split(slide_id)[-1]
            if grid_number == 0:
                print("Skipped : ", slide_name, t,' | ' ,grid_number)
                continue
                
            if self.train:
                if grid_number < 32:
                    print("Skipped : ", slide_name, t, ' | ' ,grid_number)
                    continue
            slides.append(slide_id)
            grid.append(g)
            targets.append(t)
        
        return {'slides': slides, 'grid': grid, 'targets': targets}


    def preprocess(self):
        """
            Change format of lib file to:
            {
                'slides': [xx.tif,xx2.tif , ....],
                'grid'  : [[(x,y),(x,y),..], [(x,y),(x,y),..] , ....],
                'targets': [0,1,0,1,0,1,0, etc]
            }
            len(slides) == len(grid) == len(targets)
        """
        grid    = []
        targets = []
        slides  = []
        class_names = [str(x) for x in range(len(self.classmap))]
        for i, cls_id in enumerate(class_names):
            slide_dicts = os.listdir(os.path.join(self.root, self.split, cls_id))
            print('--> | ', cls_id, ' | ', len(slide_dicts))
            for idx, slide in enumerate(slide_dicts[:self.nslides]):
                slide_folder = os.path.join(self.root, self.split, cls_id, slide)
                grid_number  = len(os.listdir(slide_folder))
                # skip empty folder
                if grid_number == 0:
                    print("Skipped : ", slide, cls_id,' | ' ,grid_number)
                    continue
                
                if self.train:
                    if grid_number < 32:
                        print("Skipped : ", slide, cls_id, ' | ' ,grid_number)
                        continue

                grid_p = []
                for id_patch in os.listdir(slide_folder):
                    grid_p.append(id_patch)

                slides.append(slide_folder)
                grid.append(grid_p)
                targets.append(int(cls_id))

        print(len(slides), len(grid), len(targets))
        return {'slides': slides, 'grid': grid, 'targets': targets}





