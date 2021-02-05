from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

import random
import cv2 
import numpy as np
import matplotlib.pyplot as plt

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix='_mask'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        #================================ Random Indexes to add noise =====================================
        self.n_idxs = int(0.2 * len(self.ids)) # Outlier = 20%
        self.idxs = random.sample(range(len(self.ids)), self.n_idxs)
        self.rng = np.random.RandomState(100)
        self.flip_prob = 0.2
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def dilation_and_erosion(self, img):
        logging.info("====== Dilating and Erosion on Images ========")

        # Taking a matrix of size 5 as the kernel 
        kernel = np.ones((5,5), np.uint8) 

        # img_erosion = cv2.erode(img, kernel, iterations=1) 
        img_dilation = cv2.dilate(img, kernel, iterations=1) 
  
        cv2.imshow('Input', img) 
        # cv2.imshow('Erosion', img_erosion) 
        cv2.imshow('Dilation', img_dilation) 
  
        cv2.waitKey(0)
        return torch.tensor(img_dilation)

    """ 
    Adding Gaussian noise to outlier masks
    """
    def add_gaussian_noise(self, img_mask):
        logging.info("====== Adding Gaussian Noise on Image Masks ========")
        # _, H, W  = img_mask.shape

        # flip = self.rng.binomial(1, self.flip_prob, size=(H, W))  # generates a mask for input

        # Mask
        # plt.imshow(img_mask.permute(1, 2, 0))
        # plt.show()

        noise = torch.FloatTensor(img_mask.shape).uniform_(0, 1)

        output_mask = img_mask * noise

        _mask = output_mask.clone()

        _mask[output_mask<0.7] = 0

        # Noisy Mask
        # plt.imshow((_mask).permute(1, 2, 0))
        # plt.show()
        return img_mask


    def __getitem__(self, i):
        print("i value:", i ,"\n")
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        if i in self.idxs:
            mask = self.add_gaussian_noise(torch.from_numpy(mask).type(torch.FloatTensor))
        else:
            mask  = torch.from_numpy(mask).type(torch.FloatTensor)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
