from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import utils
from math import atan2, asin, sin, cos
import torch.nn.utils as ut

class Transition(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, GPU=True):
        super(Transition, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm1 = nn.LSTMCell(input_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(input_dim, hidden_dim)
        self.xyz_out = nn.Linear(hidden_dim, output_dim)
        self.zeta_out = nn.Linear(hidden_dim, output_dim)
        self.uvw_out = nn.Linear(hidden_dim, output_dim)
        self.pqr_out = nn.Linear(hidden_dim, output_dim)

        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        
        self.zero = self.Tensor([[0.]])
        
    def forward(self, state_action):
        outputs = []
        h1_t = torch.zeros(1, self.hidden_dim, dtype=torch.float).cuda()
        h2_t = torch.zeros(1, self.hidden_dim, dtype=torch.float).cuda()
        
        c1_t = torch.zeros(1, self.hidden_dim, dtype=torch.float).cuda()
        c2_t = torch.zeros(1, self.hidden_dim, dtype=torch.float).cuda()
        
        H = state_action.size()[0]
        for i in range(H):
            net_input = state_action[i].unsqueeze(0)
            
            h1_t, c1_t = self.lstm1(net_input, (h1_t, c1_t))
            h2_t, c2_t = self.lstm2(net_input, (h2_t, c2_t))
            
            xyz = self.xyz_out(h1_t)
            uvw = self.uvw_out(h1_t)

            zeta = self.zeta_out(h2_t)
            pqr = self.pqr_out(h2_t)

            outputs.append(torch.cat([xyz, zeta, uvw, pqr],dim=1))
        return outputs

    def update(self, optimizer, criterion, xyzs, zetas, uvws, pqrs, actions):

        xs, ys = [], []
        H = len(xyzs)
        i = 0
        #print("Update uvw: ", uvws)
        #input("Paused")
        # process data
        for xyz, zeta, uvw, pqr in zip(xyzs, zetas, uvws, pqrs):   
            xyz_nn, zeta_nn, uvw_nn, pqr_nn = utils.numpy_to_pytorch(xyz, zeta, uvw, pqr)
            if i < H-1:
                action = actions[i].reshape((1,-1))
                action = torch.from_numpy(action).float()
                if self.GPU:
                    action = action.cuda()
                state = torch.cat([xyz_nn, zeta_nn.sin(), zeta_nn.cos(), uvw_nn, pqr_nn],dim=1)
                state_action = torch.cat([state, action],dim=1)
                xs.append(state_action)
            if i > 0:
                outputs = torch.cat([xyz_nn, zeta_nn, uvw_nn, pqr_nn],dim=1)
                ys.append(outputs)
            i += 1

        # update
        optimizer.zero_grad()
        ys_pred = self.forward(torch.stack(xs).squeeze(1))
        #print(torch.stack(ys_pred).size())
        ys_pred = torch.stack(ys_pred).squeeze(1)
        ys = torch.stack(ys).squeeze(1)
        loss = criterion(ys_pred, ys)
        #print("PREDICTED: ", ys_pred)
        #print("ACTUAL: ",ys)
        #print("LOSS: ", loss.item())
        #input("Paused")
        loss.backward()
        ut.clip_grad_norm_(self.parameters(),0.1)
        optimizer.step()
        return loss.item()