import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os

import argparse
import cv2

from dataset.utils import superpixel_segmentation
from dataset.common import *

SUPERPIXEL_DIR = 'Superpixels'

def voc_setup(root: str, image_paths: list[str], mask_paths: list[str], args: argparse.Namespace): 
    os.mkdir(os.path.join(root, SUPERPIXEL_DIR))

    for image_path, label_path in zip(image_paths, mask_paths):
        image_superpixels, label_superpixels = superpixel_segmentation(image_path, label_path, 
                                                                       seg_colors_voc[-1][0], 
                                                                       args.n_superpixels,
                                                                       args.n_iters,
                                                                       args.n_levels,
                                                                       args.n_histogram_bins,
                                                                       args.prior)
        # TODO: store superpixels (maybe using pickle? or np.save if possible)