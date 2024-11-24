import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import numpy as np
import random
import os
from PIL import Image, ImageFilter
import cv2
import pandas as pd


class DatasetFromFilenames:

    def __init__(self, filenames_loc_pos,filenames_loc_neg,transform):
        self.filenames_pos = filenames_loc_pos
        self.filenames_neg = filenames_loc_neg
        self.paths_pos = get_paths(self.filenames_pos)
        self.paths_neg = get_paths(self.filenames_neg)
        self.num_im = len(self.paths_pos)
        self.transform = transform
        

    def __len__(self):
        return len(self.paths_pos)

    def __getitem__(self, index):

        im_path_pos = self.paths_pos[index % self.num_im]
        image_pos = Image.open(im_path_pos)
        image_pos = np.asarray(image_pos).astype(np.float32) 
        im_pos = self.transform(image_pos)


        im_path_neg = self.paths_neg[index % self.num_im]
        image_neg = Image.open(im_path_neg)
        image_neg = np.asarray(image_neg).astype(np.float32) 
        im_neg = self.transform(image_neg)

        return im_pos,im_neg 


def get_paths(fname):
    paths = []
    with open(fname, 'r') as f:
        for line in f:
            temp = str(line).strip()
            paths.append(temp)
    return paths