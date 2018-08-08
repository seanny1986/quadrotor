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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.action_mu = torch.nn.Linear(hidden_dim, output_dim)
        self.action_logvar = torch.nn.Linear(hidden_dim, output_dim)
        self.value_head = torch.nn.Linear(hidden_dim, 1)
        self.gamma = params["gamma"]
        self.lmbd = params["lambda"]
        self.GPU = GPU
        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def forward(self, x):
        x = F.tanh(self.l1(x))
        mu = self.action_mu(x)
        logvar = self.action_logvar(x)
        value = self.value_head(x)
        return mu, logvar, value

    def select_action(self, x):
        mu, logvar, value = self.forward(x)
        dist = Normal(mu, logvar.exp().sqrt())
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def update(self, optim, trajectory):
        log_probs = torch.stack(trajectory["log_probs"]).float()
        rewards = torch.stack(trajectory["rewards"]).float()
        values = torch.stack(trajectory["values"]).float()
        masks = torch.stack(trajectory["dones"])
        returns = self.Tensor(rewards.size(0),1)
        deltas = self.Tensor(rewards.size(0),1)
        advantages = self.Tensor(rewards.size(0),1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i]+self.gamma*prev_return*masks[i]
            deltas[i] = rewards[i]+self.gamma*prev_value*masks[i]-values.data[i]
            advantages[i] = deltas[i]+self.gamma*self.lmbd*prev_advantage*masks[i]
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-10)
        returns = (returns-returns.mean())/(returns.std()+1e-10)
        optim.zero_grad()
        actor_loss = -(log_probs*advantages).mean()
        critic_loss = F.smooth_l1_loss(values, returns)
        loss = actor_loss+critic_loss
        loss.backward(retain_graph=True)
        optim.step()


class Trainer:
    def __init__(self, env_name, params):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.params = params
        self.action_bound = self.env.action_bound[1]
        self.trim = np.array(self.env.trim)
        self.iterations = params["iterations"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.seed = params["seed"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]
        network_settings = params["network_settings"]
        self.agent = GAE(state_dim, hidden_dim, action_dim, network_settings, GPU=cuda)
        self.optim = torch.optim.Adam(self.agent.parameters())
        self.trim = np.array(self.env.trim)
        if cuda:
            self.Tensor = torch.cuda.FloatTensor
            self.agent = self.agent.cuda()
        else:
            self.Tensor = torch.Tensor
        
        self.best = None

        # initialize experiment logging
        self.logging = params["logging"]
        if self.logging:
            self.directory = os.getcwd()
            filename = self.directory + "/data/gae-"+self.env_name+".csv"
            with open(filename, "w") as csvfile:
                self.writer = csv.writer(csvfile)
                self.writer.writerow(["episode", "reward"])
                self.train()
        else:
            self.train()

    def train(self):
        interval_avg = []
        avg = 0
        for ep in range(1, self.iterations+1):
            r_, v_, lp_, dones = [], [], [], []
            batch_mean_rwd = 0
            bsize = 1
            num_episodes = 1
            while bsize<self.batch_size+1:
                state = self.Tensor(self.env.reset())
                if ep % self.log_interval == 0 and self.render:
                    self.env.render()
                running_reward = 0
                for t in range(10000):          
                    action, log_prob, value = self.agent.select_action(state)
                    a = action.cpu().numpy()
                    next_state, reward, done, _ = self.env.step(a)
                    running_reward += reward
                    if ep % self.log_interval == 0 and self.render:
                        self.env.render()
                    next_state = self.Tensor(next_state)
                    reward = self.Tensor([reward])
                    r_.append(reward)
                    v_.append(value)
                    lp_.append(log_prob)
                    dones.append(self.Tensor([not done]))
                    if done:
                        break
                    state = next_state
                bsize += t
                batch_mean_rwd = (running_reward*(num_episodes-1)+running_reward)/num_episodes
                num_episodes += 1
            if (self.best is None or batch_mean_rwd > self.best) and self.save:
                    self.best = running_reward
                    utils.save(self.agent, self.directory + "/saved_policies/gae.pth.tar")
            trajectory = {
                        "rewards": r_,
                        "dones": dones,
                        "values": v_,
                        "log_probs": lp_}
            for _ in range(self.epochs):
                self.agent.update(self.optim, trajectory)
            interval_avg.append(batch_mean_rwd)
            avg = (avg*(ep-1)+running_reward)/ep   
            if ep % self.log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print('Episode {}\t Interval average: {:.3f}\t Average reward: {:.3f}'.format(ep, interval, avg))
                interval_avg = []
                if self.logging:
                    self.writer.writerow([ep, avg])