import random
import argparse
import torch
import policy
import environment
import mbps_utils
import copy
import csv
from torch.autograd import Variable
from mpl_toolkits.mplot3d import axes3d 
import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import memory
import datetime
import numpy as np
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch MBPS Node')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--warmup', type=int, default=100, metavar='w', help='number of warmup episodes')
parser.add_argument('--batch-size', type=int, default=64, metavar='bs', help='training batch size')
parser.add_argument('--load', default=False, type=bool, metavar='l', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', type=bool, default=True, metavar='s', help='saves the model (default is True)')
parser.add_argument('--save-epochs', type=int, default=10, metavar='ep', help='save every n epochs (default 100)')
parser.add_argument('--load-path', type=str, default='', metavar='lp', help='load path string')
parser.add_argument('--cuda', type=bool, default=True, metavar='c', help='use CUDA for GPU acceleration (default True)')
parser.add_argument('--plot-interval', type=int, default=100, metavar='pi', help='interval between plot updates')
args = parser.parse_args()

if args.cuda:
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.Tensor

dyn = mbps_utils.resume('model.pth.tar')
pol = policy.FeedForwardPolicy(state_size+goal_size, 32, action_size, dyn, args.cuda)
pol_opt = torch.optim.Adam(pol.parameters(), lr=1e-4)

with open('maneuvers.csv', 'r') as f:
  reader = csv.reader(f)
  maneuvers = list(reader)

maneuver_list = []
for i, m in enumerate(maneuvers):
    if i > 0:
        x0 = [float(x) for x in m[0:16]]
        g0 = [float(g) for g in m[16:]]
        maneuver_list.append([x0, g0])
maneuvers = maneuver_list
