# coding:utf-8
import torchvision.transforms.functional as TF
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import glob
from natsort import natsorted
import numpy as np
import cv2
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

class Med_dataset(Dataset):
    def __init__(self, data_dir, sub_dir, mode, transform):
        super(Med_dataset, self).__init__()
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'ir'))) #2_MRI
        self.length = len(self.img_names)
        self.transform = transform  #


    def __getitem__(self, index):
        img_name = self.img_names[index]
        vis = cv2.imread(os.path.join(self.root_dir, 'vi', img_name), 0)
        ir = cv2.imread(os.path.join(self.root_dir, 'ir', img_name), 0)
        vis = Image.fromarray(vis)
        ir = Image.fromarray(ir)

        vis = self.transform(vis)
        ir = self.transform(ir)

        return vis, ir, img_name

    def __len__(self):
        return self.length

class Med_testdataset(Dataset):
    def __init__(self, data_dir, sub_dir, mode, color, transform):
        super(Med_testdataset, self).__init__()
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, color)))
        self.length = len(self.img_names)
        self.transform = transform  #
        self.color = color

    def __getitem__(self, index):
        img_name = self.img_names[index]
        if self.color in ['SPECT', 'PET']:
            vis = cv2.imread(os.path.join(self.root_dir, self.color, img_name), 1)
        else:
            vis = cv2.imread(os.path.join(self.root_dir, self.color, img_name), 0)

        ir = cv2.imread(os.path.join(self.root_dir, 'MRI', img_name), 0)

        ir = Image.fromarray(ir)
        vis = Image.fromarray(vis)

        vis = self.transform(vis)
        ir = self.transform(ir)

        return vis, ir, img_name

    def __len__(self):
        return self.length
