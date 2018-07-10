import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
from collections import deque

"""
    Implements policy network class for the cross-entropy method.
"""

class CEM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, action_bound, GPU=False):
        super(CEM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.action_bound = action_bound

        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def set_weights(self, weights):
        s_size = self.input_dim
        h_size = self.hidden_dim
        a_size = self.output_dim
        
        # separate the weights for each layer
        fc1_end = (s_size*h_size)+h_size
        fc1_W = torch.from_numpy(weights[:s_size*h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size*h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(h_size*a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_size*a_size):])
        
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
    
    def get_weights_dim(self):
        return (self.input_dim+1)*self.hidden_dim+(self.hidden_dim+1)*self.output_dim
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))*self.action_bound
        return x.cpu().data