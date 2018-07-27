import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from collections import namedtuple
import copy
import random

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.__affine1 = nn.Linear(state_dim, hidden_dim)
        self.__action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.__affine1(x))
        mu = self.__action_head(x)
        return F.sigmoid(mu)

class DDPG(nn.Module):
    def __init__(self, actor, target_actor, critic, target_critic, network_settings, GPU=True, clip=None):
        super(DDPG, self).__init__() 
        self.__actor = actor
        self.__target_actor = target_actor

        self.__critic = critic
        self.__target_critic = target_critic

        self.__gamma = network_settings["gamma"]
        self.__tau = network_settings["tau"]

        self._hard_update(self.__target_actor, self.__actor)
        self._hard_update(self.__target_critic, self.__critic)

        self.__GPU = GPU
        self.__clip = clip

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.__actor = self.__actor.cuda()
            self.__target_actor = self.__target_actor.cuda()
            self.__critic = self.__critic.cuda()
            self.__target_critic = self.__target_critic.cuda()
        else:
            self.Tensor = torch.FloatTensor

    def select_action(self, state, noise=None):
        self.__actor.eval()
        with torch.no_grad():
            mu = self.__actor((Variable(state)))
        self.__actor.train()
        if noise is not None:
            sigma = Variable(torch.Tensor(noise.noise()))
            if self.__GPU:
                sigma = sigma.cuda()
            return F.sigmoid(mu+sigma).pow(0.5)
        else:
            return F.sigmoid(mu).pow(0.5)

    def random_action(self, noise):
        action = self.Tensor([noise.noise()])
        return action
    
    def _soft_update(self, target, source, tau):
	    for target_param, param in zip(target.parameters(), source.parameters()):
		    target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)
    
    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update(self, batch, crit_opt, pol_opt):
        state = Variable(torch.stack(batch.state))
        action = Variable(torch.stack(batch.action))
        with torch.no_grad():
            next_state = Variable(torch.stack(batch.next_state))
            reward = Variable(torch.cat(batch.reward))
        reward = torch.unsqueeze(reward, 1)

        next_action = self.__target_actor(next_state)                                                 # take off-policy action
        next_state_action = torch.cat([next_state, next_action],dim=1)                              # next state-action batch
        next_state_action_value = self.__target_critic(next_state_action)                             # target q-value
        with torch.no_grad():
            expected_state_action_value = (reward+(self.__gamma*next_state_action_value))             # value iteration

        crit_opt.zero_grad()                                                                   # zero gradients in optimizer
        state_action_value = self.__critic(torch.cat([state, action],dim=1))                          # zero gradients in optimizer
        value_loss = F.smooth_l1_loss(state_action_value, expected_state_action_value)              # (critic-target) loss
        value_loss.backward()                                                                       # backpropagate value loss
        crit_opt.step()                                                                        # update value function
        
        pol_opt.zero_grad()                                                                    # zero gradients in optimizer
        policy_loss = self.__critic(torch.cat([state, self.__actor(state)],1))                          # use critic to estimate pol gradient
        policy_loss = -policy_loss.mean()                                                           # sum losses
        policy_loss.backward()                                                                      # backpropagate policy loss
        if self.__clip is not None:
            torch.nn.utils.clip_grad_norm_(self.__critic.parameters(), self.__clip)                     # clip gradient
        pol_opt.step()                                                                         # update policy function
        
        self._soft_update(self.__target_critic, self.__critic, self.__tau)                                 # soft update of target networks
        self._soft_update(self.__target_actor, self.__actor, self.__tau)                                   # soft update of target networks