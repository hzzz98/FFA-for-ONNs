import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import pandas as pd


class Optical_Layer(nn.Module):
    def __init__(self, in_dim=30, out_dim=30):
        super(Optical_Layer, self).__init__()
        self.weight1 = nn.Parameter(torch.Tensor(30, 30))
        nn.init.uniform_(self.weight1, a=0, b=1)


    def forward(self, x):
        # First operation: Element-wise multiplication with weight1
        x = torch.mul(x, self.weight1)  # Element-wise multiplication
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))  # Flatten the input tensor
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)  # Normalize along the specified dimension
        return x_direction

class Postprocessing_Layer(nn.Linear):
    def __init__(self, in_dim, out_dim,bias=True, device=None, dtype=None):
        super().__init__(in_dim, out_dim, bias, device, dtype)
        self.relu = torch.nn.ReLU()

    def forward(self, x_direction):
        # Second operation: Linear transformation followed by ReLU
        x_transformed = torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        output = self.relu(x_transformed)
        return output

class Layer(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super(MyModel, self).__init__()
        self.layers = nn.ModuleList([Optical_Layer(), Postprocessing_Layer(in_dim, out_dim)])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
  