import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from collections import namedtuple
import copy
import random

class Actor(nn.Module):
    def __init__(self, action_bound, state_dim, action_dim, neurons=64):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.affine1 = nn.Linear(state_dim, neurons)
        self.action_head = nn.Linear(neurons, action_dim)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        mu = self.action_head(x)
        return self.action_bound*F.sigmoid(mu)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, neurons=64):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(state_dim+action_dim, neurons)
        self.value_head = nn.Linear(neurons, action_dim)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        q = self.value_head(x)
        return q

class DDPG(nn.Module):
    def __init__(self, actor, target_actor, critic, target_critic, tau=0.001):
        super(DDPG, self).__init__()
        self.actor = actor
        self.target_actor = target_actor

        self.critic = critic
        self.target_critic = target_critic
        self.tau = tau

        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

    def select_action(self, state, noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state, volatile=True)))
        self.actor.train()
        if noise is not None:
            sigma = Variable(torch.Tensor(noise.noise()))
            mu += sigma
        return mu

    def random_action(self, noise):
        return Variable(torch.Tensor([noise.noise()]))
    
    def soft_update(self, target, source, tau):
	    for target_param, param in zip(target.parameters(), source.parameters()):
		    target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)
    
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update(self, batch):
        state = Variable(torch.stack(torch.FloatTensor(batch.state)))
        next_state = Variable(torch.stack(torch.FloatTensor(batch.next_state)),volatile=True)
        action = Variable(torch.stack(batch.action))
        reward = Variable(torch.cat(torch.FloatTensor([batch.reward])))
        reward = torch.unsqueeze(reward, 1)
        
        next_action = self.target_actor(next_state)                                                 # take off-policy action
        next_state_action = torch.cat([next_state, next_action],dim=1)                              # next state-action batch
        next_state_action_value = self.target_critic(next_state_action)                             # target q-value
        expected_state_action_value = reward+(args.gamma*next_state_action_value)                   # value iteration

        crit_opt.zero_grad()                                                                        # zero gradients in optimizer
        state_action_value = self.critic(torch.cat([state, action],dim=1))                          # zero gradients in optimizer
        value_loss = F.smooth_l1_loss(state_action_value, expected_state_action_value)              # (critic-target) loss
        value_loss.backward()                                                                       # backpropagate value loss
        torch.nn.utils.clip_grad_norm(self.critic.parameters(),0.1)                                 # clip critic gradients
        crit_opt.step()                                                                             # update value function
        
        pol_opt.zero_grad()                                                                         # zero gradients in optimizer
        policy_loss = self.critic(torch.cat([state, self.actor(state)],1))                          # use critic to estimate pol gradient
        policy_loss = policy_loss.mean()                                                            # sum losses
        policy_loss.backward()                                                                      # backpropagate policy loss
        torch.nn.utils.clip_grad_norm(self.actor.parameters(),0.1)                                  # clip policy gradient
        pol_opt.step()                                                                              # update policy function
        
        self.soft_update(self.target_critic, self.critic, self.tau)                                 # soft update of target networks
        self.soft_update(self.target_actor, self.actor, self.tau)                                   # soft update of target networks

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])
class ReplayMemory(object):
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
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)