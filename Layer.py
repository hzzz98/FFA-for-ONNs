import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import pandas as pd


class Layer(nn.Linear): # only digital FC
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
#         self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
#         self.num_epochs = 1000
#         self.conv0 = nn.Conv2d(1, 1, kernel_size=5,padding=4,bias=False)
#         self.weight1 = nn.Parameter(torch.Tensor(30,30)) #SLM coding
#         nn.init.uniform_(self.weight1, a=0, b=1)

    def forward(self, x):
#         x = torch.mul(x,self.weight1)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)


        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

  