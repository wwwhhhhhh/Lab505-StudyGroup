import os
import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

random.seed(0)
class bile_duct_cancer(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, num=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.masks = []
        self.imgs = []
        if num is not None:
            self.imgs_lists = os.listdir(image_dir)[0:num]
            np.random.seed(0)
            np.random.shuffle(self.imgs_lists)
            self.masks_lists = os.listdir(mask_dir)[0:num]
            np.random.seed(0)
            np.random.shuffle(self.masks_lists)
        else:
            self.imgs_lists = os.listdir(image_dir)
            np.random.seed(0)
            np.random.shuffle(self.imgs_lists)
            self.masks_lists = os.listdir(mask_dir)
            np.random.seed(0)
            np.random.shuffle(self.masks_lists)
        for img in self.imgs_lists:
            img_path = os.path.join(self.image_dir, img)
            self.imgs.append(np.load(img_path, allow_pickle=True).item()['data'][:,:,:].transpose(1, 2, 0))
        for mask in self.masks_lists:
            mask_path = os.path.join(self.mask_dir, mask)
            self.masks.append(cv2.imread(mask_path, 0))


    def __len__(self):
        return len(self.masks_lists)

    def __getitem__(self, index):


        image = self.imgs[index]
        mask = self.masks[index]

        # image = np.array(Image.open(img_path).convert("RGB"))
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 215.0] = 1.0
        mask[mask == 30.0] = 0.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask, self.imgs_lists[index].split('.')[0]
