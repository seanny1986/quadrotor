import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
from collections import deque

"""
    Implements policy network class for the cross-entropy method. This should be used as a sanity
    check and benchmark for other methods, since CEM is usually embarrassingly effective.

    Credits to OpenAI for most of this code. Minor changes were made to fit in with the conventions
    used in other policy search methods in ths library, but other than that, it's mostly intact.
"""

class CEM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, GPU=False):
        super(CEM, self).__init__()
        
        # neural network dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def set_weights(self, weights):
        
        # separate the weights for each layer
        fc1_end = (self.input_dim*self.hidden_dim)+self.hidden_dim
        fc1_W = torch.from_numpy(weights[:self.input_dim*self.hidden_dim].reshape(self.input_dim, self.hidden_dim))
        fc1_b = torch.from_numpy(weights[self.input_dim*self.hidden_dim:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(self.hidden_dim*self.output_dim)].reshape(self.hidden_dim, self.output_dim))
        fc2_b = torch.from_numpy(weights[fc1_end+(self.hidden_dim*self.output_dim):])
        
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
    
    def get_weights_dim(self):
        return (self.input_dim+1)*self.hidden_dim+(self.hidden_dim+1)*self.output_dim
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x)).pow(0.5)
        return x.cpu().detach().numpy()[0]