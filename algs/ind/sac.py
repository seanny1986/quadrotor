import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gym
import gym_aero
import utils
import csv
import os
import numpy as np
from collections import namedtuple
import random


"""
PyTorch implementation of the Soft Actor-Critic algorithm (Haarnoja, 2017).
"""

class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        pass

    def forward(self, x):
        pass

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        pass
    
    def forward(self, x):
        pass

class SoftActorCritic(torch.nn.Module):
    def __init__(self, policy, soft_q_net, v_net):
        super(SoftActorCritic, self).__init__()
        self.policy = policy
        self.soft_q_net = soft_q_net
        self.v_net = v_net

    def soft_q_update(self, batch_size, 
           gamma=0.99,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=1e-2,
          ):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        expected_q_value = soft_q_net(state, action)
        expected_value   = value_net(state)
        new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)


        target_value = target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
    

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss  = std_lambda  * log_std.pow(2).mean()
        z_loss    = z_lambda    * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        soft_q_optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        
        for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward"])
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

class Trainer:
    def __init__(self):
        pass