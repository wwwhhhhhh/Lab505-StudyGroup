import os
import random
import tifffile as tiff
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from wh_frame.parameters import class_all

class forest(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, num=None, mode='train', label_to_keep=None, label_to_remove=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
        self.label_to_keep = label_to_keep
        self.label_to_remove = label_to_remove
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
            self.imgs.append(self.read_image(img_path, 'images'))
            # self.imgs.append(np.load(img_path, allow_pickle=True).item()['data'][0:50,:,:].transpose(1, 2, 0))
        for mask in self.masks_lists:
            mask_path = os.path.join(self.mask_dir, mask)
            self.masks.append(self.read_image(mask_path, 'labels'))

    def read_image(self, name, folder):
        if folder == 'images':
            image = np.asarray(tiff.imread(name))
            image = image
            image = image[np.newaxis, :, :]
            if np.isnan(image).any():
                breakpoint()
        elif folder == 'labels':
            image = tiff.imread(name)
            image = np.squeeze(image)
            image = self.change_labels(image)
            if np.isnan(image).any():
                breakpoint()
        image.flags.writeable = True
        return image

    def change_labels(self, image):
        image[image > 8] = 8
        x = np.array([], dtype=int)
        y = np.array([], dtype=int)# 存放不想要标签位置的横纵坐标
        for i in self.label_to_remove:
            pos = np.where(image == i)
            x = np.concatenate((x, pos[0]), axis=0)
            y = np.concatenate((y, pos[1]), axis=0)
        image[x, y] = 0

        class_number = 1 # 新的标签

        for i in self.label_to_keep:
            image[image == i] = class_number
            class_number += 1
        return image

    def __len__(self):
        return len(self.masks_lists)

    def __getitem__(self, index):


        image = self.imgs[index]
        mask = self.masks[index]
        if self.transform !=None:
            for func in self.transform:
               image = func(image, mask)[0]
               mask = func(image, mask)[1]

        image = np.array(image, dtype="int32").squeeze().transpose(2, 0, 1)
        label = np.array(mask, dtype="int64")

        return torch.tensor(image), torch.tensor(mask), self.imgs_lists[index].split('.')[0]



class Forest_dataset(Dataset):

    def __init__(self, map_dir, map_seffix, label_dir, label_seffix, have_label, transform=[]):
        super(Forest_dataset, self).__init__()
        map_set = []
        label_set = []
        labeltype_length = len(label_seffix)
        listfile = os.listdir(label_dir)
        for path in listfile:
            if path[(-labeltype_length):].upper() != label_seffix.upper():
                continue
            map_set.append(map_dir + path)
            label_set.append(label_dir + path)

        self.map_set = map_set
        self.label_set = label_set
        self.transform = transform
        self.is_train  = have_label
        self.n_data    = len(self.map_set)


    def read_image(self, name, folder):
        if folder =='images':
            image = tiff.imread(name)
            image = image[np.newaxis,:,:]
            if np.isnan(image).any():
                breakpoint()

        else:
            image = tiff.imread(name)
            image = np.squeeze(image)
            image[image != 3] = 0
            image[image == 3] = 1
            if np.isnan(image).any():
                breakpoint()

        image.flags.writeable = True
        return image

    def get_train_item(self, index):

        map_name = self.map_set[index]
        name = map_name.split('/')[-1]
        label_name = self.label_set[index]
        image = self.read_image(map_name, 'images')
        label = self.read_image(label_name, 'labels')
        # cv2.imwrite('./before.bmp', image[0])
        if self.transform !=None:
            for func in self.transform:
               image = func(image, label)[0]
               label = func(image, label)[1]
        # cv2.imwrite('./after.bmp', image[0])
        # cv2.imwrite('./afterlabel.bmp', label*255)
        image = np.array(image, dtype="int32").squeeze().transpose(2, 0, 1)
        label = np.array(label, dtype="int64")
        return torch.tensor(image), torch.tensor(label), name


    def get_test_item(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'images')
        # image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255

        return torch.tensor(image).squeeze(), name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def __len__(self):
        return self.n_data
