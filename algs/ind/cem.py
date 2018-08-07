import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
from collections import deque
import gym
import gym_aero
import utils
import numpy as np
from collections import deque
import csv
import os

"""
    Implements policy network class for the cross-entropy method. This should be used as a sanity
    check and benchmark for other methods, since CEM is usually embarrassingly effective.

    Credits to Udacity for most of this code. Minor changes were made to fit in with the conventions
    used in other policy search methods in this library, but other than that, it's mostly intact.
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
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x.cpu().detach().numpy()[0]


class Trainer:
    def __init__(self, env_name, params):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.action_bound = self.env.action_bound[1]
        self.trim = np.array(self.env.trim)
        self.iterations = params["iterations"]
        self.gamma = params["gamma"]
        self.seed = params["seed"]
        self.pop_size = params["pop_size"]
        self.elite_frac = params["elite_frac"]
        self.sigma = params["sigma"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]

        self.agent = CEM(state_dim, hidden_dim, action_dim, GPU=cuda)

        if cuda:
            self.Tensor = torch.cuda.FloatTensor
            self.agent = self.agent.cuda()
        else:
            self.Tensor = torch.Tensor
        
        # initialize experiment logging
        self.logging = params["logging"]
        if self.logging:
            directory = os.getcwd()
            filename = directory + "/data/cem-"+self.env_name+".csv"
            with open(filename, "w") as csvfile:
                self.writer = csv.writer(csvfile)
                self.writer.writerow(["episode", "reward"])
                self.train()
        else:
            self.train()

    def train(self):
        def evaluate(weights, rend):
            self.agent.set_weights(weights)
            episode_return = 0.0
            state = self.env.reset()
            if rend:
                self.env.render()
            for t in range(self.env.H):
                state = self.Tensor(state)
                action = self.agent(state)
                state, reward, done, _ = self.env.step(self.trim+action*5)
                if rend:
                    self.env.render()
                episode_return += reward*math.pow(self.gamma, t)
                if done:
                    break
            return episode_return
        n_elite=int(self.pop_size*self.elite_frac)
        scores_deque = deque(maxlen=100)
        best_weight = self.sigma*np.random.randn(self.agent.get_weights_dim())
        for i_iteration in range(1, self.iterations+1):
            weights_pop = [best_weight+(self.sigma*np.random.randn(self.agent.get_weights_dim())) for i in range(self.pop_size)]
            rewards = np.array([evaluate(weights, False) for weights in weights_pop])
            elite_idxs = rewards.argsort()[-n_elite:]
            elite_weights = [weights_pop[i] for i in elite_idxs]
            best_weight = np.array(elite_weights).mean(axis=0)
            if i_iteration % self.log_interval == 0:
                reward = evaluate(best_weight, True)
            else:
                reward = evaluate(best_weight, False)
            scores_deque.append(reward)
            if i_iteration % self.log_interval == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))
                if self.logging:
                    self.writer.writerow([i_iteration, np.mean(scores_deque).item()])