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
from torch.optim import lr_scheduler


data_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomRotation(degrees=(180, 180))])

params_train = {'batch_size': 256,
            'shuffle': True,
            'num_workers': 4}

params_val = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 4}
train_pos_filenames='FFA-multilayer/MNIST_layer0/pos.txt'
train_neg_filenames='FFA-multilayer/MNIST_layer0/neg.txt'
train_dataset = DatasetFromFilenames(train_pos_filenames,train_neg_filenames,data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, **params_train)

test_filenames = 'FFA-multilayer/MNIST_layer0/test.txt'
params_val2 = {'batch_size': 10,
              'shuffle': False,
              'num_workers': 4}
test_loader = torch.utils.data.DataLoader(DatasetFromFilenames_test(test_filenames,data_transform), **params_val2)


def setup_seed(seed):  
     torch.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)

device = torch.device('cuda:4') 
model_op = Optical_Layer().to(device)
model_po = Postprocessing_Layer(900,20).to(device)
print(model_op)
print(model_po)

num_epochs = 1000
num_epochs_2 = 1000
threshold=2

optimizer_op = optim.Adam(model_op.parameters(), lr=0.05)
optimizer_po = optim.Adam(model_po.parameters(), lr=0.05)

theta = torch.tensor(100).to(device)
vector = torch.normal(0,1,size=(1,10)).to(device)
# Weighted physical constraints
best_acc=[] 
test_loss_all=[] 

for epoch in range(1, num_epochs + 1): 
    train(model_op,model_po, train_loader, optimizer_op,optimizer_po, theta,vector,epoch, threshold) 
    test_acc = test_test(model_op, model_po,test_loader2,vector)

for epoch in range(num_epochs + 1, num_epochs_2 + 1): 
    train(model_op,model_po, train_loader, optimizer1, theta,vector,epoch, threshold) 
    test_acc = test_test(model_op,model_po, test_loader2,vector)

    