import torch
import random
import torch.nn.functional as F
from torch.distributions import Normal
import gym
import gym_aero
import utils
import csv
import os
import numpy as np

"""
    Pytorch implementation of Generalized Advantage Estimation (Schulman, 2015). Uses independent
    Gaussian actions (i.e. diagonal covariance matrix)
"""

class GAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, params, GPU=True):
        super(GAE,self).__init__()
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.__action_mu = torch.nn.Linear(hidden_dim, output_dim)
        self.__action_logvar = torch.nn.Linear(hidden_dim, output_dim)
        self.__value_head = torch.nn.Linear(hidden_dim, 1)
        self.__gamma = params["gamma"]
        self.__lmbd = params["lambda"]
        self.__GPU = GPU
        if GPU:
            self.__Tensor = torch.cuda.FloatTensor
        else:
            self.__Tensor = torch.Tensor

    def forward(self, x):
        x = F.tanh(self.__l1(x))
        mu = self.__action_mu(x)
        logvar = self.__action_logvar(x)
        value = self.__value_head(x)
        return mu, logvar, value

    def select_action(self, x):
        mu, logvar, value = self.forward(x)
        sigma = logvar.exp().sqrt()+1e-10
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def update(self, optim, trajectory):
        log_probs = torch.stack(trajectory["log_probs"]).float()
        rewards = torch.stack(trajectory["rewards"]).float()
        values = torch.stack(trajectory["values"]).float()
        masks = torch.stack(trajectory["dones"])
        returns = self.__Tensor(rewards.size(0),1)
        deltas = self.__Tensor(rewards.size(0),1)
        advantages = self.__Tensor(rewards.size(0),1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i]+self.__gamma*prev_return*masks[i]
            deltas[i] = rewards[i]+self.__gamma*prev_value*masks[i]-values.data[i]
            advantages[i] = deltas[i]+self.__gamma*self.__lmbd*prev_advantage*masks[i]
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-10)
        returns = (returns-returns.mean())/(returns.std()+1e-10)
        optim.zero_grad()
        actor_loss = -(log_probs.sum(dim=1, keepdim=True)*advantages).mean()
        critic_loss = F.smooth_l1_loss(values, returns)
        loss = actor_loss+critic_loss
        loss.backward(retain_graph=True)
        optim.step()


class Trainer:
    def __init__(self, env_name, params):
        self.__env = gym.make(env_name)
        self.__env_name = env_name
        self.__params = params
        self.__iterations = params["iterations"]
        self.__batch_size = params["batch_size"]
        self.__epochs = params["epochs"]
        self.__seed = params["seed"]
        self.__render = params["render"]
        self.__log_interval = params["log_interval"]
        self.__save = params["save"]
        state_dim = self.__env.observation_space.shape[0]
        action_dim = self.__env.action_space.shape[0]
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]
        network_settings = params["network_settings"]
        self.__agent = GAE(state_dim, hidden_dim, action_dim, network_settings, GPU=cuda)
        self.__optim = torch.optim.Adam(self.__agent.parameters())
        if cuda:
            self.__Tensor = torch.cuda.FloatTensor
            self.__agent = self.__agent.cuda()
        else:
            self.__Tensor = torch.Tensor
        
        self.__best = None

        # initialize experiment logging
        self.__logging = params["logging"]
        self.__directory = os.getcwd()
        if self.__logging:
            filename = self.__directory + "/data/gae-"+self.__env_name+".csv"
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
            r_, v_, lp_, dones = [], [], [], []
            batch_mean_rwd, num_episodes, bsize = 0, 1, 1
            while bsize<self.__batch_size+1:
                state = self.__env.reset()
                state = self.__Tensor(state)
                if ep % self.__log_interval == 0 and self.__render:
                    self.__env.render()
                t, running_reward = 0, 0
                done = False
                while not done:          
                    action, log_prob, value = self.__agent.select_action(state)
                    a = action.cpu().numpy()
                    next_state, reward, done, _ = self.__env.step(a)
                    running_reward += reward
                    if ep % self.__log_interval == 0 and self.__render:
                        self.__env.render()
                    next_state = self.__Tensor(next_state)
                    reward = self.__Tensor([reward])
                    r_.append(reward)
                    v_.append(value)
                    lp_.append(log_prob)
                    dones.append(self.__Tensor([not done]))
                    state = next_state
                    t += 1
                bsize += t
                batch_mean_rwd = (running_reward*(num_episodes-1)+running_reward)/num_episodes
                num_episodes += 1
            if (self.__best is None or batch_mean_rwd > self.__best) and self.__save:
                print("---Saving best GAE policy---")
                self.__best = batch_mean_rwd
                utils.save(self.__agent, self.__directory+"/saved_policies/gae-"+self.__env_name+".pth.tar")
            trajectory = {
                        "rewards": r_,
                        "dones": dones,
                        "values": v_,
                        "log_probs": lp_}
            for _ in range(self.__epochs):
                self.__agent.update(self.__optim, trajectory)
            interval_avg.append(batch_mean_rwd)
            avg = (avg*(ep-1)+running_reward)/ep   
            if ep % self.__log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print('Episode {}\t Interval average: {:.3f}\t Average reward: {:.3f}'.format(ep, interval, avg))
                interval_avg = []
                if self.__logging:
                    self.__writer.writerow([ep, interval, avg])
        utils.save(self.__agent, self.__directory + "/saved_policies/gae-"+self.__env_name+"-final.pth.tar")