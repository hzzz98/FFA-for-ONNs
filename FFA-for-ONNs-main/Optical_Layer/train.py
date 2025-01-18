import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import pandas as pd

def train(model_op, model_po,train_loader, optimizer_op,optimizer_po, theta,vector,epoch,threshold):
    model_op.train()
    model_po.train()
    for batch_idx, (data_pos,data_neg) in enumerate(train_loader):
        data_pos, data_neg = data_pos.to(device), data_neg.to(device)  
        optimizer_op.zero_grad()                     
        output_pos_op = model_op(data_pos).pow(2).mean(1)
        output_neg_op = model_op(data_neg).pow(2).mean(1)
        loss_op = torch.log(1+torch.exp(torch.cat([-output_pos_op+threshold,output_neg_op-threshold]))).mean()
        loss_op.backward()
        optimizer_op.step()
        model_op.weight1.data.clamp_(0,1)
        
    for batch_idx, (data_pos,data_neg) in enumerate(train_loader):
        data_pos, data_neg = data_pos.to(device), data_neg.to(device)  
        optimizer_po.zero_grad()     
        data_pos = model_op(data_pos)
        data_neg = model_op(data_neg)
        output_pos_post = model_po(data_pos).pow(2).mean(1)
        output_neg_post = model_po(data_neg).pow(2).mean(1)
        loss_post = torch.log(1+torch.exp(torch.cat([-output_pos_post+threshold,output_neg_post-threshold]))).mean()
        loss_post.backward()
        optimizer_po.step()

        if batch_idx % 1000 == 0:
            print(loss_post)


def test_test(model_op, model_po, test_loader,vector):
    model_op.eval()
    model_po.eval()
    correct = 0
    comparation = 0
    test_acc = 0
    with torch.no_grad():
        for data_test in test_loader:
            data_test = data_test.to(device)
            output_test = model_po(model_op(data_test)).pow(2).mean(1)
            max_values, max_indices = torch.max(output_test,dim=0)
            if max_indices == 0:
                correct = correct+1

    test_acc = 100*correct/(len(test_loader.dataset)*0.1)
    
    print(test_acc)
    return test_acc

