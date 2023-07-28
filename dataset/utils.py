import torch
import cv2
import numpy as np

def superpixel_segmentation(image_path: str, mask_path: str, mask_dontcare: int, n_superpixels: int, n_iters: int, n_levels: int, n_histogram_bins: int, prior: int) -> tuple[list[np.ndarray]]: 
    image = cv2.imread(image_path) # (H, W, C), np.ndarray, dtype=np.uint8
    mask = cv2.imread(mask_path) # (H, W, C), np.ndarray, dtype=np.uint8

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    height, width, channels = image.shape

    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, 
                                               height, 
                                               channels, 
                                               n_superpixels,
                                               n_levels, 
                                               prior, 
                                               n_histogram_bins)
    seeds.iterate(hsv_image, n_iters)

    n_superpixels_ = seeds.getNumberOfSuperpixels()
    labels: np.ndarray = seeds.getLabels() # (H, W), np.ndarray, dtype=np.int32
    
    # split superpixels into a list of superpixels
    image_superpixels = []
    mask_superpixels = []
    for i in range(n_superpixels_):
        indices = np.where(labels == i)

        top_left = (np.min(indices[1]), np.min(indices[0]))
        bottom_right = (np.max(indices[1]), np.max(indices[0]))
        
        cropped_image = image[top_left[1] : bottom_right[1] + 1, top_left[0] : bottom_right[0] + 1, :]
        cropped_mask = mask[top_left[1] : bottom_right[1] + 1, top_left[0] : bottom_right[0] + 1, :]

        cropped_label = labels[top_left[1] : bottom_right[1] + 1, top_left[0] : bottom_right[0] + 1]
        cropped_mask[np.where(cropped_label != i)] = mask_dontcare
        
        image_superpixels.append(cropped_image)
        mask_superpixels.append(cropped_mask)

    return image_superpixels, mask_superpixels

def _sample_uncertainty(preds: torch.Tensor) -> torch.Tensor:
    # preds: (N, C, H, W)
    # return: (N)

    top2_probs = torch.topk(preds, k=2, dim=1)[0] # (N, 2, H, W)
    bests = top2_probs[:, 0, :, :] # (N, H, W)
    second_bests = top2_probs[:, 1, :, :] # (N, H, W)
    best_versus_second_best = bests / second_bests # (N, H, W)
    uncertainty = torch.mean(best_versus_second_best, dim=(1, 2)) # (N)
    return uncertainty

def sample_uncertainty(preds: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    # preds: (N, C, H, W)
    # mask: (N, H, W)
    # return: (N)

    top2_probs = torch.topk(preds, k=2, dim=1)[0] # (N, 2, H, W)
    bests = top2_probs[:, 0, :, :] # (N, H, W)
    second_bests = top2_probs[:, 1, :, :] # (N, H, W)
    best_versus_second_best = bests / second_bests # (N, H, W)
    if mask is not None:
        best_versus_second_best = best_versus_second_best * mask
    uncertainty = torch.mean(best_versus_second_best, dim=(1, 2)) # (N)
    return uncertainty
