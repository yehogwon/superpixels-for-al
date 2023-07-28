import os

import torch
from torchvision.datasets import VisionDataset

from typing import Callable, Any, Optional

class SegDataset(VisionDataset): 
    def __init__(self, root: str, name: str, mode: str='train', transforms: Optional[Callable]=None, transform: Optional[Callable]=None, target_transform: Optional[Callable]=None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = root
        self.name = name
        self.mode = mode

        assert self.name in ['voc', 'city'], f'Invalid dataset name: {self.name}'
        assert self.mode in ['train', 'val'], f'Invalid dataset mode: {self.mode}'

        if name == 'voc': # self.root contains ImageSets, JPEGImages, SegmentationClass
            self._voc_init()
        elif name == 'city': # self.root contains ... (to be added)
            self._city_init()
    
    def _voc_init(self): 
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.mask_dir = os.path.join(self.root, 'SegmentationClass')

        with open(os.path.join(self.root, 'ImageSets', 'Segmentation', f'{self.mode}.txt')) as f:
            self.imaeg_ids = f.read().splitlines()
        self.image_paths = [os.path.join(self.image_dir, f'{image_id}.jpg') for image_id in self.imaeg_ids]
        self.dataset_len = len(self.image_paths)

        # TODO: implement data loading functionality efficiently

    def _city_init(self): 
        # TODO: work on cityscapes dataset
        raise NotImplementedError('Cityscapes dataset is not supported yet. Please use VOC dataset instead.')

    def __getitem__(self, index: int) -> Any:
        pass

    def __len__(self) -> int:
        return self.dataset_len
    