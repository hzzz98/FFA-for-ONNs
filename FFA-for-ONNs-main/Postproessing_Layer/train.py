import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import pandas as pd

def train(model, train_loader, optimizer, theta,vector,epoch, loss_all,threshold,pattern_block_float):
    model.train()
    for batch_idx, (data_pos,data_neg) in enumerate(train_loader):
        data_pos, data_neg = data_pos.to(device), data_neg.to(device)  
      
        optimizer.zero_grad()                     
        output_pos = model(data_pos).pow(2).mean(1)
        output_neg = model(data_neg).pow(2).mean(1)
        
        loss = torch.log(1+torch.exp(torch.cat([-output_pos+threshold,output_neg-threshold]))).mean()

        
        loss.backward() # local loss for update the gradient of the trainable layer
        optimizer.step() # Only digital FC


        if batch_idx % 1000 == 0:
            print(loss)

            
def test(model, test_loader,vector):
    model.eval()
   
    correct = 0
    comparation = 0
    test_acc = 0

    with torch.no_grad():
        
        for data_test in test_loader:
            data_test = data_test.to(device)
            output_test = model(data_test).pow(2).mean(1)
            max_values, max_indices = torch.max(output_test,dim=0)
            if max_indices == 0:
                correct = correct+1
    test_acc = 100*correct/(len(test_loader.dataset)*0.1)
    
    print(test_acc)
    return test_acc
