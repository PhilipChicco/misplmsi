import os
import sys
import logging
import argparse
import glob
import yaml

import numpy as np
import openslide
import cv2
import json
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from wsi_tools.utils import get_files


def run(wsi_path, level, npy_path, json_path):
    if not os.path.isfile(json_path):
        print(f'MISSING {json_path}')
        return

    # get the level * dimensions e.g. tumor0.tif level 6 shape (1589, 7514)
    slide = openslide.OpenSlide(wsi_path)
    w, h  = slide.level_dimensions[level]
    mask_tumor = np.zeros((h, w))  # the init mask, and all the value is 0

    # get the factor of level * e.g. level 6 is 2^6
    factor = slide.level_downsamples[level]

    with open(json_path) as f:
        dicts = json.load(f)
    tumor_polygons = dicts['positive']

    for tumor_polygon in tumor_polygons:
        # plot a polygon
        name = tumor_polygon["name"]
        vertices = np.array(tumor_polygon["vertices"]) / factor
        vertices = vertices.astype(np.int32)

        cv2.fillPoly(mask_tumor, [vertices], (255))

    mask_tumor = mask_tumor[:] > 127
    mask_tumor = np.transpose(mask_tumor)

    # load tissue mask and ensure white spaces match
    # assumes tissue mask is in same directory for npy saving
    npy_tissue = npy_path.replace('_tumor.npy', '_tissue.npy')
    tissue_mask = np.load(npy_tissue)
    assert tissue_mask.shape == mask_tumor.shape
    mask_tumor[tissue_mask == 0] = 0

    np.save(npy_path, mask_tumor)
    npy_png = npy_path.replace('.npy', '.png')
    plt.imsave(npy_png, mask_tumor.transpose(), vmin=0, vmax=1, cmap='gray')
    


def main(args):
    logging.basicConfig(level=logging.INFO)

    wsi_dir  = args['tumor_mask']['wsi_dir']
    wsi_text = args['tumor_mask']['wsi_text']
    npy_dir  = args['tumor_mask']['npy_dir']
    level    = args['tumor_mask']['level']
    wsi_ext  = args['tumor_mask']['wsi_ext']
    json_dir = args['tumor_mask']['json_dir']

    if wsi_dir:
        wsi_dir = glob.glob(os.path.join(wsi_dir, wsi_ext))
    elif wsi_text:
        wsi_text = get_files(wsi_text, wsi_ext)
        wsi_dir  = wsi_text
    else:
        print('SELECT AN OPTION')
        return

    # create a npy save dir if inexistant
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    #root_dir = "/media/philipchicco/2.2/Anonymized_Img/"

    for wsi_file in wsi_dir:
        wsi_name = (os.path.split(wsi_file)[-1]).split(".")[0]
        npy_file  = os.path.join(npy_dir, wsi_name + "_tumor.npy")
        json_file = os.path.join(json_dir, wsi_name + ".json")
        
        if not os.path.isfile(json_file): continue
        print(wsi_name, wsi_file)
        run(wsi_file, level, npy_file, json_file)
    print()
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                                 ' it in npy format')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="research_mil/configs/amc_2020/wsi_tools_config_amc.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    main(cfg)

