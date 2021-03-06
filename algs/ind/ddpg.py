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
Implementation of deterministic policy gradient. This implementation directly outputs both the mean and logvariance
of the policy, rather than using exploratory noise (OU-process, as used in the paper). I've had trouble getting this
working even for simple tasks like hover. It did -- at one stage -- work for Mujoco tasks, but seems to not learn
any more, so it seems like it's very sensitive to the hyperparameters used.
"""

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.__affine1 = nn.Linear(state_dim, hidden_dim)
        self.__action_head = nn.Linear(hidden_dim, action_dim)
        self.__logvar_head = nn.Linear(hidden_dim, action_dim)
        self.__action_head.weight.data.mul_(0.1)
        self.__action_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.__affine1(x))
        mu = self.__action_head(x)
        logvar = self.__logvar_head(x)
        return mu, logvar

class DDPG(nn.Module):
    def __init__(self, actor, critic, target_critic, network_settings, GPU=True, clip=None):
        super(DDPG, self).__init__()
        self.__actor = actor
        self.__critic = critic
        self.__target_critic = target_critic
        self.__gamma = network_settings["gamma"]
        self.__tau = network_settings["tau"]
        self._hard_update(self.__target_critic, self.__critic)
        self.__GPU = GPU
        self.__clip = clip
        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.__actor = self.__actor.cuda()
            self.__critic = self.__critic.cuda()
            self.__target_critic = self.__target_critic.cuda()
        else:
            self.Tensor = torch.FloatTensor

    def select_action(self, state, deterministic=False):
        mu, logvar = self.__actor(state)
        if deterministic:
            return mu
        else:
            sigma = logvar.exp().sqrt()+1e-10
            dist = Normal(mu, sigma)
            return dist.sample()

    def random_action(self, noise):
        action = self.Tensor(noise.noise())
        return action

    def _soft_update(self, target, source, tau):
	    for target_param, param in zip(target.parameters(), source.parameters()):
		    target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)

    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update_critic(self, opt, batch):
        states = torch.stack(batch.state)
        actions = torch.stack(batch.action)
        with torch.no_grad():
            next_states = torch.stack(batch.next_state)
            rewards = torch.cat(batch.reward)
        rewards = torch.unsqueeze(rewards, 1)
        next_actions = self.select_action(next_states)                               # take on-policy action
        next_state_actions = torch.cat([next_states, next_actions],dim=1)                              # next state-action batch
        next_state_action_values = self.__target_critic(next_state_actions)                           # target q-value
        with torch.no_grad():
            expected_state_action_values = rewards+self.__gamma*next_state_action_values      # value iteration
        opt.zero_grad()                                                                        # zero gradients in optimizer
        state_action_values = self.__critic(torch.cat([states, actions],dim=1))                        # zero gradients in optimizer
        value_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)              # (critic-target) loss
        value_loss.backward()
        if self.__clip is not None:
            torch.nn.utils.clip_grad_norm_(self.__critic.parameters(), self.__clip)                                                                        # backpropagate value loss
        opt.step()
        opt.zero_grad() 
        self._soft_update(self.__target_critic, self.__critic, self.__tau)

    def update_policy(self, opt, batch):
        states = torch.stack(batch.state)                                                                     # update value function
        opt.zero_grad()                                                                         # zero gradients in optimizer
        policy_loss = self.__critic(torch.cat([states, self.select_action(states)], dim=1))                      # use critic to estimate pol gradient
        policy_loss = -policy_loss.mean()                                                           # sum losses
        policy_loss.backward()                                                                      # backpropagate policy loss
        if self.__clip is not None:
            torch.nn.utils.clip_grad_norm_(self.__actor.parameters(), self.__clip)                 # clip gradient
        opt.step()                                                                              # update policy function
        

class Trainer:
    def __init__(self, env_name, params):
        # initialize environment
        self.__env = gym.make(env_name)
        self.__env_name = env_name

        # save important experiment parameters for the training loop
        self.__iterations = params["iterations"]
        self.__mem_len = params["mem_len"]
        self.__seed = params["seed"]
        self.__render = params["render"]
        self.__log_interval = params["log_interval"]
        self.__warmup = params["warmup"]
        self.__batch_size = params["batch_size"]
        self.__learning_updates = params["learning_updates"]
        self.__save = params["save"]

        # initialize DDPG agent using experiment parameters from config file
        state_dim = self.__env.observation_space.shape[0]
        action_dim = self.__env.action_space.shape[0]
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]
        network_settings = params["network_settings"]
        actor = Actor(state_dim, hidden_dim, action_dim)
        critic = utils.Critic(state_dim+action_dim, hidden_dim, 1)
        target_critic = utils.Critic(state_dim+action_dim, hidden_dim, 1)
        self.__agent = DDPG(actor,
                        critic,
                        target_critic,
                        network_settings,
                        GPU=cuda)

        # intitialize ornstein-uhlenbeck noise for random action exploration
        ou_scale = params["ou_scale"]
        ou_mu = params["ou_mu"]
        ou_sigma = params["ou_sigma"]
        self.__noise = utils.OUNoise(action_dim, scale=ou_scale, mu=ou_mu, sigma=ou_sigma)
        self.__noise.set_seed(self.__seed)
        self.__memory = ReplayMemory(self.__mem_len)
        self.__pol_opt = torch.optim.Adam(actor.parameters(), params["actor_lr"])
        self.__crit_opt = torch.optim.Adam(critic.parameters(),params["critic_lr"])

        # want to save the best policy
        self.__best = None

        # send to GPU if flagged in experiment config file
        if cuda:
            self.__Tensor = torch.cuda.FloatTensor
            self.__agent = self.__agent.cuda()
        else:
            self.__Tensor = torch.Tensor

        # initialize experiment logging. This wipes any previous file with the same name
        self.__logging = params["logging"]
        self.__directory = os.getcwd()
        if self.__logging:
            filename = self.__directory + "/data/ddpg-"+self.__env_name+".csv"
            with open(filename, "w") as csvfile:
                self.__writer = csv.writer(csvfile)
                self.__writer.writerow(["episode", "interval", "reward"])
                self._run_algo()
        else:
            self._run_algo()

    def _run_algo(self):
        interval_avg = []
        avg = 0
        for ep in range(1, self.__iterations+1):

            state = self.__env.reset()
            state = self.__Tensor(state)
            self.__noise.reset()
            running_reward = 0
            if ep % self.__log_interval == 0 and self.__render:
                self.__env.render()
            done = False
            while not done:

                # select an action using either random policy or trained policy
                if ep < self.__warmup:
                    action = self.__agent.random_action(self.__noise).data
                else:
                    action = self.__agent.select_action(state).data

                # step simulation forward
                next_state, reward, done, _ = self.__env.step(action.cpu().numpy())
                running_reward += reward

                # render the episode if render selected
                if ep % self.__log_interval == 0 and self.__render:
                    self.__env.render()

                # transform to tensors before storing in memory
                next_state = self.__Tensor(next_state)
                reward = self.__Tensor([reward])

                # push to replay memory
                self.__memory.push(state, action, next_state, reward)

                # online training if out of warmup phase
                if ep >= self.__warmup:
                    for i in range(5):
                        transitions = self.__memory.sample(self.__batch_size)
                        batch = Transition(*zip(*transitions))
                        self.__agent.update_critic(self.__crit_opt, batch)
                    for i in range(3):
                        transitions = self.__memory.sample(self.__batch_size)
                        batch = Transition(*zip(*transitions))
                        self.__agent.update_policy(self.__pol_opt, batch)

                # step to next state
                state = next_state

            if (self.__best is None or running_reward > self.__best) and ep > self.__warmup and self.__save:
            #if ep % self.__log_interval == 0:
                self.__best = running_reward
                print("---Saving best DDPG policy---")
                utils.save(self.__agent, self.__directory + "/saved_policies/ddpg-"+self.__env_name+".pth.tar")

            # anneal noise
            if ep > self.__warmup:
                self.__noise.anneal()

            # print running average and interval average, log average to csv file
            interval_avg.append(running_reward)
            avg = (avg*(ep-1)+running_reward)/ep
            if ep % self.__log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print("Episode {}\t Interval average: {:.3f}\t Average reward: {:.3f}".format(ep, interval, avg))
                interval_avg = []
                if self.__logging:
                    self.__writer.writerow([ep, interval, avg])
        utils.save(self.__agent, self.__directory + "/saved_policies/ddpg-"+self.__env_name+"-final.pth.tar")


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
