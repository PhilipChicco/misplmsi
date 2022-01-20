import os
import argparse
import logging
import glob
import cv2
import yaml
import sys

from PIL import Image
import numpy as np
import openslide
from matplotlib import pyplot as plt
from histomicstk.saliency.tissue_detection import get_tissue_mask

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from wsi_tools.utils import get_files

def get_tissue(wsi_image_thumbnail,th=1,mz=15):
    
    wsi_image_thumbnail_copy = wsi_image_thumbnail.copy()

    hsv_image = cv2.cvtColor(wsi_image_thumbnail, cv2.COLOR_RGB2HSV)
    
    _, rgbbinary    = cv2.threshold(hsv_image[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rgbbinary       = rgbbinary.astype("uint8")
    kernel          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    rgbbinary_close = cv2.morphologyEx(rgbbinary, cv2.MORPH_CLOSE, kernel)
    rgbbinary_open  = cv2.morphologyEx(rgbbinary_close, cv2.MORPH_OPEN, kernel)
    #
    contours, _  = cv2.findContours(rgbbinary_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_on_wsi = cv2.drawContours(
        wsi_image_thumbnail_copy, contours, -1, (0, 255, 0), 5)

    tissue = np.zeros((wsi_image_thumbnail.shape[0],wsi_image_thumbnail.shape[1]),np.uint8)
    t_mask = cv2.fillPoly(tissue, pts=contours, color=(255,255,255))
    t_mask = np.clip(t_mask,0,1)
    #
    mask_tissue = get_tissue_mask(
                    wsi_image_thumbnail, deconvolve_first=True,
                    n_thresholding_steps=th, sigma=0., min_size=mz)[0]
    tissue_mask = np.clip(mask_tissue,0,1)
    tissue_mask[t_mask == 0] = 0
    return tissue_mask, contours_on_wsi


def run(wsi_path, level, npy_path):
    
    slide = openslide.OpenSlide(wsi_path)

    img_RGB = np.array(slide.read_region((0, 0),
                       level,
                       slide.level_dimensions[level]).convert('RGB'))
         
    img_RGB[img_RGB.copy() == 0] = 255
    img_RGB = np.uint8(img_RGB)

    # Remove inks: optional
    # img_RGB = detect_blue(img_RGB, 'black')
    # img_RGB = detect_blue(img_RGB, 'blue')
    # img_RGB = detect_blue(img_RGB, 'green')
    # img_RGB = detect_blue(img_RGB, 'red')

    # best
    tissue_mask, contours_on_wsi = get_tissue(img_RGB,1)

    np.save(npy_path, tissue_mask.transpose())
    npy_png = npy_path.replace('.npy', '.png')
    plt.imsave(npy_png, tissue_mask, vmin=0, vmax=1, cmap='gray')
    npy_png = npy_png.replace('.png', '_fig.png')
    plt.imsave(npy_png, img_RGB)
    npy_png = npy_png.replace('.png', '_cnts.png')
    plt.imsave(npy_png, contours_on_wsi)

    # Sanity check
    # Uncomment to view 
    # plt.subplot(1,3,1)
    # plt.imshow(img_RGB)
    # plt.subplot(1,3,2)
    # plt.imshow(contours_on_wsi)
    # plt.subplot(1,3,3)
    # plt.imshow(tissue_mask,vmin=0, vmax=1, cmap='gray')
    # plt.show()

def detect_blue(image_orig, remove='blue'):
    image = cv2.cvtColor(image_orig, cv2.COLOR_RGB2BGR)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define blue color range
    if remove == 'blue':
        light_blue = np.array([100,50,50]) 
        dark_blue  = np.array([140,255,255]) 
    elif remove == 'red':# red
        light_blue = np.array([161,155,84]) 
        dark_blue  = np.array([180,255,255]) 
    # green 
    elif remove == 'green':# red
        light_blue = np.array([ 25, 52,72]) 
        dark_blue  = np.array([102,255,255])
    else: # black
        light_blue = np.array([0,  0, 0]) 
        dark_blue  = np.array([179,100,130])
    

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, light_blue, dark_blue)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(image,image, mask= mask)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    #
    img_RGB_B = np.clip(output,0,1)
    image_orig[img_RGB_B == 1] = 255
    return image_orig


def main(args):

    wsi_dir  = args['tissue_mask']['wsi_dir']
    wsi_text = args['tissue_mask']['wsi_text']
    npy_dir  = args['tissue_mask']['npy_dir']
    level    = args['tissue_mask']['level']
    wsi_ext  = args['tissue_mask']['wsi_ext']

    if wsi_dir:
        wsi_dir = glob.glob(os.path.join(wsi_dir, wsi_ext))
    elif wsi_text:
        wsi_text = get_files(wsi_text, wsi_ext)
        wsi_dir  = wsi_text
    else:
        print('SELECT AN OPTION')
        return

    # create a npy save dir if inexistant
    print(f'SIZE {len(wsi_dir)}')
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    for wsi_file in wsi_dir:
        wsi_name = (os.path.split(wsi_file)[-1]).split(".")[0]
        npy_file = os.path.join(npy_dir, wsi_name + "_tissue.npy")

        if os.path.isfile(wsi_file):
            print(wsi_name, npy_file)
            run(wsi_file, level, npy_file)
        else:
            print(wsi_name, 'no image files.')# use full for mrxs 
        
    print("done!")


if __name__ == '__main__':
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
