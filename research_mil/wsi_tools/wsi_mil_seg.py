"""
    Script to create the whole-slide objects
"""
import warnings

warnings.filterwarnings("ignore")

import os, copy, glob
import argparse, yaml
import random, sys, math
from tqdm import tqdm
import numpy as np
import openslide

import torch, cv2
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

from histomicstk.saliency.tissue_detection import get_tissue_mask
from utils_augmentation import HistoNormalize
from tissue_mask import get_tissue

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# import libs related to project here
from models import get_model
from loaders.datasets import GridWSIPatchDataset
from research_mil.utils.post_seg import get_scores_mil
from research_mil.utils.misc import AverageMeter, convert_state_dict


#
# Special case for loader 
class libLoader(Dataset):
    
    def __init__(self, 
                libraryfile=None,
                class_map={'mss': 0, 'msi': 1},
                nslides=-1):

        
        self.classmap = class_map
        self.nslides  = nslides

        if libraryfile:
            lib = torch.load(libraryfile)
            """ Format
               {'class_name':  torch_files.append({
                    "slide": wsi_file,
                    "grid" : points,
                    "target" : class_id }),
                 'class_name': ......,
                  ......................
                }

            """
            print('Loaded | ', libraryfile, self.classmap)
            lib = self.preprocess(lib)
    
        else:
            raise ('Please provide a lib file.')
        

        self.slides      = lib['slides']
        self.targets     = lib['targets']
        self.slide_paths = lib['slides']

    def preprocess(self, lib):
        """
            Change format of lib file to:
            {
                'slides': [xx.tif,xx2.tif , ....],
                'grid'  : [[(x,y),(x,y),..], [(x,y),(x,y),..] , ....],
                'targets': [0,1,0,1,0,1,0, etc]
            }
            len(slides) == len(grid) == len(targets)

            ## TO DO: Change the root folder.
        """
        slides = []
        grid = []
        targets = []
        class_names = [x for x in self.classmap]
        for i, cls_id in enumerate(class_names):
            slide_dicts = lib[cls_id]
            print('--> | ', cls_id, ' | ', len(slide_dicts))
            for idx, slide in enumerate(slide_dicts[:self.nslides]):

                if not slide['slide'] in slides:
                    slides.append(slide['slide'])
                    grid.append(slide['grid'])
                    targets.append(self.classmap[slide['target']])

        print(len(slides), len(grid), len(targets))
        return {'slides': slides, 'grid': grid, 'targets': targets}

def get_wsicompressed_test_lib(cfg):

    lib = os.path.join(cfg['data']['data_path'],'test_lib.pth')
    slide_test = libLoader(libraryfile=lib, class_map=cfg['data']['classmap'],
                  nslides=cfg['data']['nslides'])

    return slide_test

def make_dataloader(wsi_path, mask_path, full=False, mag_level=2, stride=0.5, cfg=None):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transform_test = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])

    batch_size  = 128
    num_workers = 6

    subtype  = cfg['subtype']
    img_size = 512 if subtype else 256

    wsi_loader = GridWSIPatchDataset(
        wsi_path=wsi_path,
        mask_path=mask_path,
        image_size=img_size,  # default : 256,  512 (mss/msi detection)
        patch_size=256,  # default : 256 - mask size 
        crop_size=256,
        normalize=True,
        full=full,
        transform=transform_test,
        level=mag_level,
        stride=stride)

    dataloader = DataLoader(wsi_loader,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=False)

    return dataloader

def get_probs_map(cfg, model, dataloader, channels=512):
    m_shape = dataloader.dataset._mask.shape
    map     = np.zeros((m_shape[0], m_shape[1], channels))
    with torch.no_grad():
        for (data, x_mask, y_mask) in tqdm(dataloader):
            image = data.cuda()
            probs = model(image)        
            probs_numpy = probs.data.cpu().numpy()
            map[x_mask, y_mask, :] = probs_numpy
    return map

def make_tissue_mask_v2(slide, level=6, thrs=1):
    thumbnail_rgb = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')
    thumbnail_rgb = np.array(thumbnail_rgb)
    thumbnail_rgb[thumbnail_rgb.copy() == 0] = 255
    thumbnail_rgb = np.uint8(thumbnail_rgb)

    mask_deconv = get_tissue(thumbnail_rgb,1)[0]
    return mask_deconv, thumbnail_rgb

def run_process_subtype(cfg, test_dset, model, save_logdir, channels=512, mag_level=2, mask_level=6, using_mask=False, full=False):
    
    slide_names = test_dset.slides
    targets     = test_dset.targets
    slide_paths = test_dset.slide_paths
    thr         = cfg['testing']['threshold']

    for idx, (slide_name, target) in enumerate(zip(slide_names, targets)):
        
        wsi_name = os.path.split(slide_name)[-1].split(".")[0]

        save_pth = os.path.join(save_logdir, str(target))
        os.makedirs(save_pth, exist_ok=True)
        
        slide = slide_paths[idx]
        slide = openslide.OpenSlide(slide)
        print()
        print('--- Working on slide |[{}/{}][tumor_mask:{}] | '.format(idx + 1, len(targets),using_mask), wsi_name, target)

        thumbnail_mask, rgb_img = make_tissue_mask_v2(slide, level=mask_level, thrs=2)
        mask      = thumbnail_mask.transpose()
        mask_size = mask.shape

        if using_mask:
            # using tumor mask
            mask_path = cfg['testing']['masks'] + '/' + wsi_name + '_tumor.png'
            mask_t    = np.array(Image.open(mask_path).convert('L'))
            mask_t    = mask_t.transpose()
            mask_t    = np.clip(np.round(mask_t), 0, 1)
            mask[mask_t == 0] = 0
        

        data_loader = make_dataloader(slide, mask, mag_level=mag_level, stride=0.5, full=full, cfg=cfg) 
        
        start_time = datetime.now()
        # probs_map
        map = get_probs_map(cfg, model, data_loader, channels)
        map = cv2.resize(np.transpose(map.copy(), axes=[1, 0, 2]), mask_size, interpolation=cv2.INTER_NEAREST)    
        end_time = datetime.now()
        print("\nInference time: %.1f minutes" % ((end_time - start_time).seconds / 60,))

        # save probability predictions
        sv_name = os.path.join(save_pth, wsi_name +'.npy')
        np.save(sv_name, map)
        plt.imsave(sv_name.replace('.npy','_heat.png'),map, cmap='jet',dpi=300)


        # prediction | selected threshold
        cam_pred = map.copy()
        cam_pred[mask.transpose() == 0] = 0.0
        cam_pred[cam_pred < thr ] = 0.0
        cam_pred[cam_pred >= thr] = 1.0 
        pred_ws = np.clip(cam_pred.copy(), 0, 1).astype(np.int)

        img = rgb_img.copy().astype(np.float32) / 255.0
        img[:, :, 0][pred_ws > 0] = 5
        img[:, :, 1][pred_ws > 0] = 0
        img[:, :, 2][pred_ws > 0] = 0
        pred_overlay = Image.fromarray(np.uint8(img * 255))
        pred_overlay.save(sv_name.replace('.npy','_ovr.png'),dpi=(300,300),quality=50,compress=9)

        img = rgb_img.copy().astype(np.float32) / 255.0
        img[:, :, 0][mask.transpose() > 0] = 1
        img[:, :, 1][mask.transpose() > 0] = 1
        img[:, :, 2][mask.transpose() > 0] = 0
        pred_overlay = Image.fromarray(np.uint8(img * 255))
        pred_overlay.save(sv_name.replace('.npy','_msk.png'),dpi=(300,300),quality=50,compress=9)

def run_process_orig(cfg, test_dset, model, save_logdir, channels=512, mag_level=2, mask_level=6, full=False):
    
    thr         = cfg['testing']['threshold']
    thr_values  = list(cfg['testing']['threshold_list'].split(","))
    dice_scores = {str(i):AverageMeter() for i in thr_values }
    iou_scores  = {str(i):AverageMeter() for i in thr_values }
    acc_scores  = {str(i):AverageMeter() for i in thr_values }
    score_list  = {str(i):[] for i in thr_values }

    slide_names = test_dset.slides
    targets     = test_dset.targets
    slide_paths = test_dset.slide_paths
    #norm_image  = HistoNormalize()

    my_cm = matplotlib.cm.get_cmap('jet')

    for idx, (slide_name, target) in enumerate(zip(slide_names, targets)):
        
        if target == 0: continue # only tumor slides

        wsi_name = os.path.split(slide_name)[-1].split(".")[0]

        if os.path.exists(os.path.join(save_logdir, wsi_name + '_grid.png')): continue

        
        slide = slide_paths[idx]
        slide = openslide.OpenSlide(slide)
        
        print(f'\r--- Working on slide |[{idx + 1}/{len(targets)}] | {wsi_name} {target}\n', end= ' ')

        thumbnail_mask, rgb_img = make_tissue_mask_v2(slide, level=mask_level, thrs=1)
        mask      = thumbnail_mask.transpose()
        mask_size = mask.shape
        

        data_loader = make_dataloader(slide, mask, mag_level=mag_level, stride=0.5, full=full, cfg=cfg)  

        ##########################################################
        start_time = datetime.now()
        # probs_map
        map = get_probs_map(cfg, model, data_loader, channels)
        
        # rescale attention to 0.0 ~ 1.0
        # if you are using a sigmoid attention, then comment this
        map = (map - np.min(map))/(np.max(map) - np.min(map))
        map = cv2.resize(np.transpose(map.copy(), axes=[1, 0, 2]), mask_size, interpolation=cv2.INTER_NEAREST)

        end_time = datetime.now()
        print("\rInference time: %.1f minutes" % ((end_time - start_time).seconds / 60), end=' ')
        ##########################################################

        # save probability predictions
        os.makedirs(os.path.join(save_logdir,'preds'), exist_ok=True)
        save_pth = os.path.join(save_logdir, 'preds', wsi_name+'.npy')
        np.save(save_pth, map)


        # ground-truth tumor masks
        mask_path  = cfg['testing']['masks'] + '/' + wsi_name + '_tumor.png'
        mask_gt    = Image.open(mask_path).convert('L')
        mask_label = np.array(mask_gt)
        mask_label[mask.transpose() == 0] = 0

        img  = rgb_img.astype(np.float32) / 255.0
        img[:, :, 0][mask_label > 0] = 0
        img[:, :, 1][mask_label > 0] = 5
        img[:, :, 2][mask_label > 0] = 0
        mask_gt = np.array(Image.fromarray(np.uint8(img * 255)))


        ##########################################################
        for th_val in thr_values:
            cam_pred = map.copy()
            cam_pred[mask.transpose() == 0] = 0.0
            cam_pred[cam_pred < float(th_val) ] = 0.0
            cam_pred[cam_pred >= float(th_val)] = 1.0  

            if np.sum(cam_pred) > 0:  # avoid zero division
                dice, iou, acc = get_scores_mil(mask_label.astype(np.int), cam_pred)
            else:
                dice, iou, acc = 0.0, 0.0, 0.0
            
            dice_scores[str(th_val)].append(dice)
            iou_scores[str(th_val)].append(iou)
            acc_scores[str(th_val)].append(acc)
            score_list[str(th_val)].append((slide_name,dice,iou,acc))

        ##########################################################

        # prediction | selected threshold
        cam_pred = map.copy()
        cam_pred[mask.transpose() == 0] = 0.0
        cam_pred[cam_pred < thr ] = 0.0
        cam_pred[cam_pred >= thr] = 1.0 
        pred_ws = np.clip(cam_pred.copy(), 0, 1).astype(np.int)

        img = rgb_img.copy().astype(np.float32) / 255.0
        img[:, :, 0][pred_ws > 0] = 5
        img[:, :, 1][pred_ws > 0] = 0
        img[:, :, 2][pred_ws > 0] = 0
        pred_blend = np.array(Image.fromarray(np.uint8(img * 255)))

        plt.figure()
        cam_n = map.copy()
        cam_n[mask.transpose() == 0] = 0.0
        cam_n = my_cm(cam_n)
        cam_n = np.uint8(255 * cam_n)
        cam_n    = Image.fromarray(cam_n).convert('RGB')
        cam_n_np = np.array(cam_n)
        
        inp_i     = torch.from_numpy(rgb_img.transpose(2, 0, 1))
        inp_m     = torch.from_numpy(mask_gt.transpose(2, 0, 1))
        cam_n_np  = torch.from_numpy(cam_n_np.transpose(2, 0, 1))
        cam_pred  = torch.from_numpy(pred_blend.transpose(2, 0, 1))

        input_grid = make_grid(inp_i.float(), nrow=1).unsqueeze_(0)
        input_mask = make_grid(inp_m.float(), nrow=1).unsqueeze_(0)
        cam_pred   = make_grid(cam_pred.float(), nrow=1).unsqueeze_(0)
        cam_hot    = make_grid(cam_n_np.float(), nrow=1).unsqueeze_(0)

        ## save individual preds
        # INPUT | GT | PRED | HEAT
        os.makedirs(os.path.join(save_logdir,'preds'), exist_ok=True)
        save_pth = os.path.join(save_logdir ,'preds', wsi_name+'_grid.png')
        cat_i = torch.cat([input_grid, input_mask, cam_pred, cam_hot], 0)
        save_image(cat_i, fp=save_pth, scale_each=True, normalize=True)
        ##########################################################

    del my_cm
    
    print()
    # save predictions results
    fp2 = open(os.path.join(save_logdir, 'all_scores.txt'), 'w')
    for th in thr_values:
        fp = open(os.path.join(save_logdir, f'predictions_seg_{th}.csv'), 'w')
        fp.write('file,dice,iou,acc\n')
        for item in score_list[str(th)]:
            fp.write('{},{:.4f},{:.4f},{:.4f}\n'.format(item[0], item[1],item[2], item[3]))
        fp.write('Average,{:.4f},{:.4f},{:.4f}\n'.format(dice_scores[str(th)].avg(),iou_scores[str(th)].avg(),acc_scores[str(th)].avg()))
        fp.close()
        print('--- (test) | THRESHOLD: {} | DICE {:.4f} | mIOU {:.4f} | pACC {:.4f} '.format(th,
        dice_scores[str(th)].avg(),iou_scores[str(th)].avg(),acc_scores[str(th)].avg()))
        fp2.write('--- (test) | THRESHOLD: {} | DICE {:.4f} | mIOU {:.4f} | pACC {:.4f} \n'.format(th,
            dice_scores[str(th)].avg(),iou_scores[str(th)].avg(),acc_scores[str(th)].avg()))
    fp2.close()

###
# This is the original method used for MIL segmentation with normal and tumor classes
# Use other main_subtype for subtyping problem
# Please ensure the magnification and mask sizes are correct
# AMC|COLON : default is 256 and 512 for patches --> resized to 256
# FOR Tumor seg: default should be 256 in dataloader
def main(cfg):
    print(cfg)
    print()

    # create root if not existing
    save_root = os.path.join(cfg['root'], cfg['testing']['logdir'])
    os.makedirs(save_root, exist_ok=True)

    # create Slide Dataset
    print('Setting up data ...')
    

    slide_dataset = get_wsicompressed_test_lib(cfg)


    print('Done Loading slides....')

    # Use pretrained model
    model = get_model(cfg)
    ckpt  = os.path.join(cfg['root'],cfg['testing']['checkpoint'])
    model_chk = torch.load(ckpt)
    state = convert_state_dict(model_chk["model_state"])
    model.load_state_dict(state)
    print(f'Loaded state_dict for {ckpt}')
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    model.eval()
    print(model.module.pooling)
    # set model to return only probs or attention
    model.module.pooling.seg_mode = 1

    subtype   = cfg['subtype']
    full_mode = cfg['full']
    print()
    print(f'RUNNING | ---> SUBTYPE {subtype} || FULL_MODE {full_mode}')
    print()
    start_time = datetime.now()
    if subtype:
        run_process_subtype(cfg, slide_dataset, model, save_root, channels=1, mag_level=0, mask_level=6, using_mask=True, full=full_mode)
    else:
        run_process_orig(cfg, slide_dataset, model, save_root, channels=1, mag_level=0, full=full_mode)
    end_time = datetime.now()
    print("\nTotal time: %.1f minutes" % ((end_time - start_time).seconds / 60,))



if __name__ == "__main__":
    # get configs
    parser = argparse.ArgumentParser(description="Segmention MIL Methods")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./research_mil/configs/amc_2020/wsi_seg.yml",
        help="Configuration file to use"
    )
    # AMC MSI Seg
    # AMC mult (mag):2 || lvl: 0
    
    parser.add_argument('--split', type=str, default='test', help='Split to process')
    parser.add_argument('--full',   action='store_true', default=False, help='Segment WSI with large tissue/tumor mask')
    parser.add_argument('--subtype',action='store_true', default=False, help='MSS/MSI segmentation ')

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    cfg['split']   = args.split
    cfg['full']    = args.full
    cfg['subtype'] = args.subtype

    main(cfg)

