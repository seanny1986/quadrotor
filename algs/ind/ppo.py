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
        super(ActorCritic,self).__init__()
        self.__l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.__mu = torch.nn.Linear(hidden_dim, output_dim)
        self.__logvar = torch.nn.Linear(hidden_dim, output_dim)
        self.__value = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.tanh(self.__l1(x))
        mu = self.__mu(x)
        logvar = self.__logvar(x)
        value = self.__value(x)
        return mu, logvar, value

class PPO(torch.nn.Module):
    def __init__(self, pi, beta, network_settings, GPU=True):
        super(PPO,self).__init__()
        self.__pi = pi
        self.__beta = beta
        self._hard_update(self.__beta, self.__pi)
        self.__gamma = network_settings["gamma"]
        self.__lmbd = network_settings["lambda"]
        self.__eps = network_settings["eps"]
        self.__GPU = GPU
        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.__pi = self.__pi.cuda()
            self.__beta = self.__beta.cuda()
        else:
            self.Tensor = torch.Tensor

    def select_action(self, x):
        mu, logvar, value = self.__beta(Variable(x))
        std = logvar.exp().sqrt()+1e-4
        a = Normal(mu, std)
        action = a.sample()
        logprob = a.log_prob(action)
        return F.tanh(action), logprob, value

    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def set_theta_old(self):
        self._hard_update(self.__beta, self.__pi)

    def update(self, optim, trajectory):
        states = torch.stack(trajectory["states"]).float()
        actions = torch.stack(trajectory["actions"]).float()
        beta_log_probs = torch.stack(trajectory["log_probs"]).float()
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
        return dist_pi.entropy().mean(dim=0).detach()


class Trainer:
    def __init__(self, env_name, params):
        self.env = gym.make(env_name)
        self.params = params
        self.iterations = params["iterations"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.seed = params["seed"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]
        self.action_bound = self.env.action_bound[1]
        hidden_dim = params["hidden_dim"]
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        cuda = params["cuda"]
        network_settings = params["network_settings"]
        learning_rate = params["lr"]
        pi = ActorCritic(state_dim, hidden_dim, action_dim)
        beta = ActorCritic(state_dim, hidden_dim, action_dim)
        self.agent = PPO(pi, beta, network_settings, GPU=cuda)
        self.optim = torch.optim.Adam(pi.parameters(), lr=learning_rate)
        self.trim = np.array(self.env.trim)
        self.best = None
        if cuda:
            self.Tensor = torch.cuda.FloatTensor
            self.agent = self.agent.cuda()
        else:
            self.Tensor = torch.Tensor

        # initialize experiment logging
        self.logging = params["logging"]
        if self.logging:
            self.directory = os.getcwd()
            filename = self.directory + "/data/ppo.csv"
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
            s_, a_, ns_, r_ = [], [], [], []
            v_, lp_, dones = [], [], []
            batch_mean_rwd = 0
            bsize = 1
            num_episodes = 1
            while bsize<self.batch_size+1:
                state = self.Tensor(self.env.reset())
                if ep % self.log_interval == 0 and self.render:
                    self.env.render()
                running_reward = 0
                for t in range(1, self.env.H+1):          
                    action, log_prob, value = self.agent.select_action(state)
                    a = self.trim+action[0].cpu().numpy()*5
                    next_state, reward, done, _ = self.env.step(a)
                    running_reward += reward
                
                    if ep % self.log_interval == 0 and self.render:
                        self.env.render()

                    next_state = self.Tensor(next_state)
                    reward = self.Tensor([reward])

                    s_.append(state[0])
                    a_.append(action[0])
                    ns_.append(next_state[0])
                    r_.append(reward)
                    v_.append(value[0])
                    lp_.append(log_prob[0])
                    dones.append(self.Tensor([not done]))
                    if done:
                        break
                    state = next_state
                bsize += (t-1)
                batch_mean_rwd = (running_reward*(num_episodes-1)+running_reward)/num_episodes
                num_episodes += 1
            if (self.best is None or batch_mean_rwd > self.best) and self.save:
                    self.best = running_reward
                    utils.save(self.agent, self.directory + "/saved_policies/gae.pth.tar")
            trajectory = {"states": s_,
                        "actions": a_,
                        "next_states": ns_,
                        "rewards": r_,
                        "dones": dones,
                        "values": v_,
                        "log_probs": lp_}
            entropy = []
            for _ in range(self.epochs):
                entropy.append(self.agent.update(self.optim, trajectory))
            #print("---Policy Entropy: {}".format(torch.stack(entropy).mean(dim=0).cpu().numpy()))
            self.agent.set_theta_old()
            interval_avg.append(batch_mean_rwd)
            avg = (avg*(ep-1)+running_reward)/ep   
            if ep % self.log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
                interval_avg = []
                if self.logging:
                    self.writer.writerow([ep, avg])

        

        