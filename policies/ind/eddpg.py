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
        self.affine1 = nn.Linear(state_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        mu = self.action_head(x)
        return F.sigmoid(mu)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        q = self.value_head(x)
        return q

class DDPG(nn.Module):
    def __init__(self, actor, target_actor, critic_1, critic_2, target_critic, action_bound, network_settings, GPU=True, clip=None):
        super(DDPG, self).__init__() 
        self.actor = actor
        self.target_actor = target_actor

        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic = target_critic

        self.action_bound = action_bound

        self.gamma = network_settings["gamma"]
        self.tau = network_settings["tau"]

        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic_1)
        self.hard_update(self.critic_2, self.critic_1)

        self.pol_opt = optim.Adam(actor.parameters(), lr=1e-4)
        self.crit_opt_1 = optim.Adam(critic_1.parameters(), lr=1e-4)
        self.crit_opt_2 = optim.Adam(critic_1.parameters(), lr=1e-4)

        self.GPU = GPU
        self.clip = clip

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.actor = self.actor.cuda()
            self.target_actor = self.target_actor.cuda()
            self.critic_1 = self.critic_1.cuda()
            self.critic_2 = self.critic_2.cuda()
            self.target_critic = self.target_critic.cuda()
        else:
            self.Tensor = torch.FloatTensor

    def select_action(self, state, noise=None):
        self.actor.eval()
        with torch.no_grad():
            mu = self.actor((Variable(state)))
        self.actor.train()
        if noise is not None:
            sigma = Variable(torch.Tensor(noise.noise()))
            if self.GPU:
                sigma = sigma.cuda()
            return F.sigmoid(mu+sigma).pow(0.5)
        else:
            return F.sigmoid(mu).pow(0.5)

    def random_action(self, noise):
        action = self.Tensor([noise.noise()])
        return action
    
    def soft_update(self, target, source, tau):
	    for target_param, param in zip(target.parameters(), source.parameters()):
		    target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)
    
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update_critic(self, critic, optim, batch):
        state = Variable(torch.stack(batch.state))
        action = Variable(torch.stack(batch.action))
        with torch.no_grad():
            next_state = Variable(torch.stack(batch.next_state))
            reward = Variable(torch.cat(batch.reward))
        reward = torch.unsqueeze(reward, 1)

        next_action = self.target_actor(next_state)                                                 # take off-policy action
        next_state_action = torch.cat([next_state, next_action],dim=1)                              # next state-action batch
        next_state_action_value = self.target_critic(next_state_action)                             # target q-value
        with torch.no_grad():
            expected_state_action_value = (reward+(self.gamma*next_state_action_value))             # value iteration

        optim.zero_grad()                                                                   # zero gradients in optimizer
        state_action_value = critic(torch.cat([state, action],dim=1))                          # zero gradients in optimizer
        value_loss = F.smooth_l1_loss(state_action_value, expected_state_action_value)              # (critic-target) loss
        value_loss.backward()                                                                       # backpropagate value loss
        optim.step()                                                                        # update value function
        
    def update_policy(self, optim, batch):
        state = Variable(torch.stack(batch.state))
        self.pol_opt.zero_grad()                                                                    # zero gradients in optimizer
        policy_loss_1 = self.critic_1(torch.cat([state, self.actor(state)],1))                          # use critic to estimate pol gradient
        policy_loss_2 = self.critic_2(torch.cat([state, self.actor(state)],1))
        policy_loss = -(policy_loss_1+policy_loss_2).mean()                                                           # sum losses
        policy_loss.backward()                                                                      # backpropagate policy loss
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip)
        self.pol_opt.step()                                                                         # update policy function
        
        self.soft_update(self.target_critic, self.critic_1, self.tau)                                 # soft update of target networks
        self.soft_update(self.target_critic, self.critic_2, self.tau)
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