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
from torch.autograd import Variable

"""
    PyTorch implementation of Proximal Policy Optimization (Schulman, 2017).
"""

class ActorCritic(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.__l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.__action_mu = torch.nn.Linear(hidden_dim, output_dim)
        self.__action_logvar = torch.nn.Linear(hidden_dim, output_dim)
        self.__value_head = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.tanh(self.__l1(x))
        mu = self.__action_mu(x)
        logvar = self.__action_logvar(x)
        value = self.__value_head(x)
        return mu, logvar, value

class PPO(torch.nn.Module):
    def __init__(self, pi, beta, params, GPU=True):
        super(PPO, self).__init__()
        self.__pi = pi
        self.__beta = beta
        self.hard_update()
        self.__gamma = params["gamma"]
        self.__lmbd = params["lambda"]
        self.__eps = params["eps"]
        self.__GPU = GPU
        if GPU:
            self.__Tensor = torch.cuda.FloatTensor
            self.__pi = self.__pi.cuda()
            self.__beta = self.__beta.cuda()
        else:
            self.__Tensor = torch.Tensor

    def select_action(self, x):
        mu, logvar, value = self.__pi(x)
        sigma = logvar.exp().sqrt()+1e-10
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def hard_update(self):
        target, source = self.__beta, self.__pi
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update(self, optim, trajectory):
        states = torch.stack(trajectory["states"]).float()
        actions = torch.stack(trajectory["actions"]).float()
        beta_log_probs = torch.stack(trajectory["log_probs"]).float()
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
        mu_pi, logvar_pi, _ = self.__pi(states)
        dist_pi = Normal(mu_pi, logvar_pi.exp().sqrt())
        pi_log_probs = dist_pi.log_prob(actions)
        ratio = (pi_log_probs-beta_log_probs.detach()).sum(dim=1, keepdim=True).exp()
        optim.zero_grad()
        actor_loss = -torch.min(ratio*advantages, torch.clamp(ratio, 1-self.__eps, 1+self.__eps)*advantages).mean()
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
        hidden_dim = params["hidden_dim"]
        state_dim = self.__env.observation_space.shape[0]
        action_dim = self.__env.action_space.shape[0]
        cuda = params["cuda"]
        network_settings = params["network_settings"]
        learning_rate = params["lr"]
        pi = ActorCritic(state_dim, hidden_dim, action_dim)
        beta = ActorCritic(state_dim, hidden_dim, action_dim)
        self.__agent = PPO(pi, beta, network_settings, GPU=cuda)
        self.__optim = torch.optim.Adam(pi.parameters(), lr=learning_rate)
        self.__best = None
        if cuda:
            self.__Tensor = torch.cuda.FloatTensor
            self.__agent = self.__agent.cuda()
        else:
            self.__Tensor = torch.Tensor

        # initialize experiment logging
        self.__logging = params["logging"]
        self.__directory = os.getcwd()
        if self.__logging:
            filename = self.__directory + "/data/ppo-"+self.__env_name+".csv"
            with open(filename, "w") as csvfile:
                self.__writer = csv.writer(csvfile)
                self.__writer.writerow(["episode", "reward"])
                self._run_algo()
        else:
            self._run_algo()

    def _run_algo(self):
        interval_avg = []
        avg = 0
        for ep in range(1, self.__iterations+1):
            s_, a_, ns_, r_ = [], [], [], []
            v_, lp_, dones = [], [], []
            batch_mean_rwd = 0
            bsize = 1
            num_episodes = 1
            while bsize<self.__batch_size+1:
                state = self.__Tensor(self.__env.reset())
                if ep % self.__log_interval == 0 and self.__render:
                    self.__env.render()
                running_reward = 0
                done = False
                t = 0
                while not done:          
                    action, log_prob, value = self.__agent.select_action(state)
                    a = action.cpu().numpy()
                    next_state, reward, done, _ = self.__env.step(a)
                    running_reward += reward
                    if ep % self.__log_interval == 0 and self.__render:
                        self.__env.render()
                    next_state = self.__Tensor(next_state)
                    reward = self.__Tensor([reward])
                    s_.append(state)
                    a_.append(action)
                    ns_.append(next_state)
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
                print("---Saving best PPO policy---")
                self.__best = batch_mean_rwd
                utils.save(self.__agent, self.__directory + "/saved_policies/ppo-"+self.__env_name+".pth.tar")
            trajectory = {"states": s_,
                        "actions": a_,
                        "next_states": ns_,
                        "rewards": r_,
                        "dones": dones,
                        "values": v_,
                        "log_probs": lp_}
            for _ in range(self.__epochs):
                self.__agent.update(self.__optim, trajectory)
            self.__agent.hard_update()
            interval_avg.append(batch_mean_rwd)
            avg = (avg*(ep-1)+batch_mean_rwd)/ep   
            if ep % self.__log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print('Episode {}\t Interval average: {:.3f}\t Average reward: {:.3f}'.format(ep, interval, avg))
                interval_avg = []
                if self.__logging:
                    self.__writer.writerow([ep, avg])

        