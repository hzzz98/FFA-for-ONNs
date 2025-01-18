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



class DatasetFromFilenames_test:

    def __init__(self, filenames_loc_test,transform):
        self.filenames_test= filenames_loc_test
        self.paths_test = get_paths(self.filenames_test)
        self.num_im = len(self.paths_test)
        self.transform = transform

        

    def __len__(self):
        return len(self.paths_test)

    def __getitem__(self, index):

        im_path_test = self.paths_test[index % self.num_im]
        image_test = Image.open(im_path_test)
        image_test = np.asarray(image_test).astype(np.float32) 
        im_test = self.transform(image_test)
        
        return im_test
                    
def get_paths(fname):
    paths = []
    with open(fname, 'r') as f:
        for line in f:
            temp = str(line).strip()
            paths.append(temp)
    return paths