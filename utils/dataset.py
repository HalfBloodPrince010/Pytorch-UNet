from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms

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
        
        self.n_idxs = 0
        self.idxs = []
        self.transform = transforms.Compose([transforms.Resize((128,128))])
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def set_outlier_indices(self, train_data_indices, outlier_percent):
        print("==Setting Outlier Indicies==\n%=", outlier_percent)
        self.n_idxs = int(outlier_percent * len(train_data_indices)) # Outlier %
        if self.n_idxs != 0:
            self.idxs = random.sample(train_data_indices, self.n_idxs)


    @classmethod
    def preprocess(cls, pil_img, scale, transform=None):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        
        # == Reducing Image Dimension ==
        if transform is not None:
            pil_img = transform(pil_img)
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        
        return img_trans

    def morph(self, img, morph_type):
   
        # Taking a matrix of size 5 as the kernel 
        kernel = np.ones((5,5), np.uint8) 
        if morph_type == 'erosion':
            img_ = cv2.erode(img, kernel, iterations=7)
        if morph_type == 'smoothing':
            img_e = cv2.erode(img, kernel, iterations=5)
            img_ = cv2.dilate(img_e, kernel, iterations=3)

        return torch.from_numpy(img_).type(torch.FloatTensor)

    """ 
    Adding Gaussian noise to outlier masks
    """
    def add_gaussian_noise(self, img_mask):
        # _, H, W  = img_mask.shape

        # flip = self.rng.binomial(1, self.flip_prob, size=(H, W))  # generates a mask for input

        noise = torch.FloatTensor(img_mask.shape).uniform_(0, 1)

        output_mask = img_mask * noise

        _mask = output_mask.clone()

        _mask[output_mask<0.7] = 0

        return img_mask

    def flip_mask_bits(self, img_mask):

        """
        Flipping the 1s to 0s and 0s to 1s
        in the Image Mask for  x % of training images
        40% - Here, around 2035 Images
        """

        res_mask = img_mask.clone()

        res_mask[img_mask == 0] = 1

        res_mask[img_mask == 1] = 0

        return res_mask


    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        
        assert len(mask_file) == 1, \
            f"Either no mask or multiple masks found for the ID {idx}: {mask_file}"
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, self.transform)
        mask = self.preprocess(mask, self.scale, self.transform)
        
        if len(self.idxs)!=0 and i in self.idxs:
            #mask = self.flip_mask_bits(torch.from_numpy(mask).type(torch.FloatTensor))
            mask = self.morph(mask, morph_type='smoothing')
        else:
            mask  = torch.from_numpy(mask).type(torch.FloatTensor)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
