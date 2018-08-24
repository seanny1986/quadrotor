import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd
import random
import numpy as np
from collections import namedtuple

"""
    This module contains utility functions and classes that would clutter up other scripts. These include getting
    trajectory information from the dataset, doing Euler angle quaternion conversions, saving and loading models,
    and logging functionality. If you need some kind of visualization of utility function to be implemented and it
    doesn't make sense for it to go anywhere else, this is where it belongs.
"""

style.use('seaborn-white')

def numpy_to_pytorch(xyz, zeta, uvw, pqr, cuda=True):
    xyz = torch.from_numpy(xyz.T).float()
    zeta = torch.from_numpy(zeta.T).float()
    uvw = torch.from_numpy(uvw.T).float()
    pqr = torch.from_numpy(pqr.T).float()
    if cuda:
        xyz = xyz.cuda()
        zeta = zeta.cuda()
        uvw = uvw.cuda()
        pqr = pqr.cuda()
    return xyz, zeta, uvw, pqr

def get_trajectories(df, state_dim, action_dim, batchsize, H, dt):
    seq_len = int(H/dt)
    states = np.zeros((seq_len, batchsize, state_dim+1))
    actions = np.zeros((seq_len, batchsize, action_dim))
    next_states = np.zeros((seq_len, batchsize, state_dim))
    data = df.loc[df['len'] == seq_len]
    for i in range(batchsize):
        initial = data.sample(n=1)
        key = initial[['key']].values
        sequence = df.loc[df['key'] == key[0][0]]
        states[:,i,:] = sequence[['X0','Y0','Z0','ROLL0','PITCH0','YAW0','U0','V0','W0','Q01','Q02','Q03','O01','O02','O03','O04','dt']].values
        actions[:,i,:] = sequence[['A01','A02','A03','A04']].values
        next_states[:,i,:] = sequence[['X1','Y1','Z1','ROLL1','PITCH1','YAW1','U1','V1','W1','Q11','Q12','Q13','O11','O12','O13','O14']].values
    return states, actions, next_states

def euler_angle_to_quaternion(roll, pitch, yaw):
    cr = math.cos(roll*0.5)
    sr = math.sin(roll*0.5)
    cp = math.cos(pitch*0.5)
    sp = math.sin(pitch*0.5)
    cy = math.cos(yaw*0.5)
    sy = math.sin(yaw*0.5)
    q_w = cy*cr*cp+sy*sr*sp
    q_x = cy*sr*cp-sy*cr*sp
    q_y = cy*cr*sp+sy*sr*cp
    q_z = sy*cr*cp-cy*sr*sp
    return [q_x, q_y, q_z, q_w]

def quaternion_to_euler_angle(x, y, z, w):
	ysqr = y*y
	t0 = +2.0*(w*x+y*z)
	t1 = +1.0-2.0*(x*x+ysqr)
	roll = math.atan2(t0,t1)
	t2 = +2.0*(w*y-z*x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	pitch = math.asin(t2)
	t3 = +2.0*(w*z+x*y)
	t4 = +1.0-2.0*(ysqr+z*z)
	yaw = math.atan2(t3,t4)
	return roll, pitch, yaw

def average_gradient(model):
    mean = []
    for param in model.parameters():
        mean.append(param.grad.mean().item())
    return float(sum(mean))/float(len(mean))

def print_gradients(model):
    for param in model.parameters():
        print(param.grad)

def save(model, filename):
    print("=> Saving model in '{}'".format(filename))
    torch.save(model, filename)

def load(filename):
    print("=> Loading '{}'".format(filename))
    return torch.load(filename, map_location=lambda storage, loc: storage)

def resume(model):
    print("=> loading model '{}'".format(model))
    return torch.load(model)

def progress(count, total, loss):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))
    percent = round(100.0 * count / float(total), 1)
    loss = tuple([round(x, 5) if isinstance(x, float) else x for x in loss])
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    print('[{}] {}%, Loss: {}'.format(bar, percent, loss), end='\r', flush=True)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Logger:
    def __init__(self, title, xlab, ylab):
        self.fig1 = plt.figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title(title)
        self.ax1.set_xlabel(xlab)
        self.ax1.set_ylabel(ylab)
        self.fig1.subplots_adjust(hspace=0.3)
        self.fig1.subplots_adjust(wspace=0.3)
        #self.fig1.tight_layout()
        self.fig1.show()

        self.legend = None

        self.ax1_data = [[],[]]
        self.ax1_val = [[],[]]

        self.model_counter = 0
        self.title = title
        self.xlab = xlab
        self.ylab = ylab

    def plot_graphs(self):
        self.ax1.clear()
        if not self.legend == None:
            self.legend.remove()
        p5, p6 = self.ax1.plot(self.ax1_data[:][0],self.ax1_data[:][1],self.ax1_val[:][0],self.ax1_val[:][1])

        self.ax1.set_title(self.title)
        self.ax1.set_xlabel(self.xlab)
        self.ax1.set_ylabel(self.ylab)
        
        self.legend = self.fig1.legend((p5, p6), ('Train', 'Validation'))

        self.fig1.canvas.draw()

    def update_data(self, J, J_val):
        i = self.model_counter
        self.ax1_data[0].append(i)
        self.ax1_data[1].append(J)
        self.ax1_val[0].append(i)
        self.ax1_val[1].append(J_val)
        self.model_counter += 1

    def save_figure(self, name):
        self.fig1.savefig(name + '.pdf', bbox_inches='tight')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def cuda_if(torch_object, cuda):
    return torch_object.cuda() if cuda else torch_object

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension)*self.mu
        self.reset()
        self.alpha = 0.01

    def reset(self):
        self.state = np.ones(self.action_dimension)*self.mu

    def noise(self):
        x = self.state
        dx = self.theta*(self.mu-x)+self.sigma*np.random.randn(len(x))
        self.state = x+dx
        return self.state*self.scale

    def set_seed(self,seed):
        np.random.seed(seed=seed)
    
    def anneal(self):
        # annealing the exploration noise by progressively stepping mu and sigma to 0. The reason
        # for returning the mean and sigma is so that I can check the determinism of the noise that
        # is being injected. If the noise is within a certain threshold, we probably don't want use 
        # it, and instead let the policy act deterministically.
        
        if abs(self.mu) > 0:
            d_mu = 0-self.mu
            d_sig = 0-self.sigma
            self.mu += self.alpha*d_mu
            self.sigma += self.alpha*d_sig
        return self.mu, self.sigma

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1)%self.capacity

    def sample(self, batch_size):
        if self.__len__() < batch_size:
            return self.memory
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor,self).__init__()
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim

        self.__l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.__mu = torch.nn.Linear(hidden_dim, output_dim)
        self.__logvar = torch.nn.Linear(hidden_dim, output_dim)
        self.__mu.weight.data.mul_(0.1)
        self.__mu.bias.data.mul_(0.0)
        self.__logvar.weight.data.mul_(0.1)
        self.__logvar.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.__l1(x))
        mu = self.__mu(x)
        logvar = self.__logvar(x)
        return mu, logvar

class ActorLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorLSTM,self).__init__()
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim

        self.__l1 = torch.nn.LSTM(input_dim, hidden_dim)
        self.__mu = torch.nn.Linear(hidden_dim, output_dim)
        self.__logvar = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        x, hidden = inputs
        x = x.view(x.size(0), -1)
        x, hidden = self.__l1(x, hidden)
        mu = self.__mu(x)
        logvar = self.__logvar(x)
        return mu, logvar, hidden

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.__affine1 = nn.Linear(input_dim, hidden_dim)
        self.__value_head = nn.Linear(hidden_dim, output_dim)
        self.__value_head.weight.data.mul_(0.1)
        self.__value_head.bias.data.mul_(0.0)
        

    def forward(self, x):
        x = F.relu(self.__affine1(x))
        q = self.__value_head(x)
        return q
    
