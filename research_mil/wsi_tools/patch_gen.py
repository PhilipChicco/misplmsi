
import warnings
warnings.filterwarnings("ignore")

import argparse, os
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader import MILdataset


def main(args):
    # TCGA mult  : 1,0, resize: False | Tissue level 2
    # amc | sev, mult: 2,0 | resize : True | 
    # REMEMBER use mult 2, level 0 for SUBTYING PATCHES on AMC|SEV [Highest level]
    # FOR TUMOR TRAINING AND HEATMAPS, use mult 1 and level 2 

    split    = args.split
    nslides  = args.nslides 
    norm_img = True
    resize   = True #False  #if m == 2
    rand_stain = False

    print(f'SPLIT {split} NSLIDES {nslides} per class. ')
    
    # AMC (Tumor Train Test VAL) 
    # Train : 240 eadch, Test: 173, Val: 92 each
    # class_map = {'normal':0,'msi':1}
    #lib     = "./research_mil/data/amc_2020/weak/tumor_normal/{}_lib.pth".format(split)
    #root    = "/media/philipchicco/CHICCO4TB/Development/projects/milresearch_logs/amc_2020/logs/amc/weak/tumor_normal_m2_l0/"

    # Example.
    #lib     = "./research_mil/data/amc_2020/weak/sev_test/{}_lib.pth".format(split)
    #root    = "/media/philipchicco/CHICCO4TB/Development/projects/milresearch_logs/amc_2020/logs/amc/weak/sev_test_m1_l2/"

    # AMC
    lib     = "/media/philipchicco/CHICCO4TB/Development/Data/MSSMSI/AMCData/AMC_LIBS/{}_lib.pth".format(split)
    root    = "/media/philipchicco/CHICCO4TB/Development/Data/MSSMSI/AMCData/patches_m2_l0"

    savedir = os.path.join(root, split)


    dset = MILdataset(libraryfile=lib, mult=2, level=0,
                      transform=None, class_map={'mss':0, 'msi':1},
                      nslides=nslides, savedir=savedir, norm_img=norm_img, resize=resize, rand_stain=rand_stain)
    loader_t = DataLoader(dset, batch_size=1, num_workers=4, shuffle=False, pin_memory=False)

    pbar = tqdm(loader_t, ncols=80, desc=' ')

    for i, _ in enumerate(pbar):
        pbar.set_description(' [{}] | [{}] :'.format(i,len(loader_t.dataset)))


if __name__ == '__main__':
    # get configs
    parser = argparse.ArgumentParser(description="Patch generation args")
    parser.add_argument("--split",  type=str, default="test", help="Split to use. ")
    parser.add_argument("--nslides",type=int, default=100000, help="Number of slides (default: 100000) ")

    args = parser.parse_args()

    main(args)