import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os

import argparse

import path

from utils import superpixel_segmentation
from common import *

import pickle
from tqdm import tqdm

def voc_setup(args: argparse.Namespace): 
    if not os.path.exists(os.path.join(args.root, args.superpixel_dir)):
        os.mkdir(os.path.join(args.root, args.superpixel_dir))
        os.mkdir(os.path.join(args.root, args.superpixel_dir, 'Images'))
        os.mkdir(os.path.join(args.root, args.superpixel_dir, 'Masks'))
    
    file_names = []
    with open(os.path.join(args.root, 'ImageSets', 'Segmentation', f'{args.imageset}.txt'), 'r') as f:
        file_names = f.readlines()
    file_names = [file_name.strip() for file_name in file_names]

    image_paths = [os.path.join(args.root, 'JPEGImages', f'{file_name}.jpg') for file_name in file_names]
    mask_paths = [os.path.join(args.root, 'SegmentationClass', f'{file_name}.png') for file_name in file_names]

    for image_path, mask_path in tqdm(zip(image_paths, mask_paths), desc='Segmenting Images/Masks into Superpixels', total=len(image_paths)):
        image_name = path.get_file_name(image_path)
        mask_name = path.get_file_name(mask_path)

        image_superpixels, mask_superpixels = superpixel_segmentation(image_path, mask_path, 
                                                                       seg_colors_voc[-1][0], 
                                                                       args.n_superpixels,
                                                                       args.n_iters,
                                                                       args.n_levels,
                                                                       args.n_histogram_bins,
                                                                       args.prior)

        with open(os.path.join(args.root, args.superpixel_dir, 'Images', f'{image_name}.pickle'), 'wb') as f:
            pickle.dump(image_superpixels, f)
        with open(os.path.join(args.root, args.superpixel_dir, 'Masks', f'{mask_name}.pickle'), 'wb') as f:
            pickle.dump(mask_superpixels, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to VOC root directory (contains JPEGImages, SegmentationClass, and ImageSets)')
    parser.add_argument('--superpixel_dir', type=str, default='Superpixels', help='Directory name to save superpixel images and masks (relative to root)')
    parser.add_argument('--imageset', type=str, default='train', help='Image set to use (One of the followings: train, val, and trainval)')
    parser.add_argument('--n_superpixels', type=int, default=100, help='Number of superpixels to use')
    parser.add_argument('--n_iters', type=int, default=10, help='Number of iterations to use for superpixel segmentation')
    parser.add_argument('--n_levels', type=int, default=5, help='Number of levels to use for superpixel segmentation')
    parser.add_argument('--n_histogram_bins', type=int, default=25, help='Number of histogram bins to use for superpixel segmentation')
    parser.add_argument('--prior', type=int, default=3, help='Prior to use for superpixel segmentation')
    args = parser.parse_args()

    voc_setup(args)
