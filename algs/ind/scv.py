import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal
import gym
import gym_aero
import utils
import csv
import os
import numpy as np
from collections import namedtuple
import random
import scipy.optimize

"""
    An attempt at an online-offline version of the Stein Control Variate version of PPO.
"""

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.__affine1 = nn.Linear(state_dim, hidden_dim)
        self.__action_head = nn.Linear(hidden_dim, action_dim)
        self.__logvar = nn.Linear(hidden_dim, action_dim)
        self.__action_head.weight.data.mul_(0.1)
        self.__action_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.__affine1(x))
        mu = self.__action_head(x)
        logvar = self.__logvar(x)
        return mu, logvar

class SCV(nn.Module):
    def __init__(self, actor, critic, target_critic, network_settings, GPU=True, clip=None):
        super(SCV, self).__init__()
        self.__actor = actor
        self.__critic = critic
        self.__target_critic = target_critic
        self.__gamma = network_settings["gamma"]
        self.__tau = network_settings["tau"]
        self.__eps = network_settings["eps"]
        self._hard_update(self.__target_critic, self.__critic)
        self.__GPU = GPU
        self.__clip = clip
        if GPU:
            self.__Tensor = torch.cuda.FloatTensor
            self.__actor = self.__actor.cuda()
            self.__target_actor = self.__target_actor.cuda()
            self.__critic = self.__critic.cuda()
            self.__target_critic = self.__target_critic.cuda()
        else:
            self.__Tensor = torch.FloatTensor

    def select_action(self, state):
        mu, logvar = self.__actor((Variable(state)))
        sigma = logvar.exp().sqrt()+1e-10
        dist = Normal(mu, sigma)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def _soft_update(self, target, source, tau):
	    for target_param, param in zip(target.parameters(), source.parameters()):
		    target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)

    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def online_update(self, opt, batch):
        state = Variable(torch.stack(batch.state))
        action = Variable(torch.stack(batch.action))
        with torch.no_grad():
            next_state = Variable(torch.stack(batch.next_state))
            reward = Variable(torch.cat(batch.reward))
            done = Variable(torch.cat(batch.done))
        reward = torch.unsqueeze(reward, 1)
        done = torch.unsqueeze(done, 1)

        next_action, _ = self.__actor(next_state)
        next_state_action = torch.cat([next_state, next_action],dim=1)
        next_state_action_value = self.__target_critic(next_state_action)
        with torch.no_grad():
            expected_state_action_value = reward+self.__gamma*next_state_action_value*(1-done)
        state_action_value = self.__critic(torch.cat([state, action],dim=1))   
        value_loss = F.smooth_l1_loss(state_action_value, expected_state_action_value)
        opt.zero_grad()
        value_loss.backward()
        opt.step()
        self._soft_update(self.__target_critic, self.__critic, self.__tau)
        
    def offline_update(self, opt, trajectory):
        states = torch.stack(trajectory["states"]).float()
        actions = torch.stack(trajectory["actions"]).float()
        beta_log_probs = torch.stack(trajectory["log_probs"]).float()
        rewards = torch.stack(trajectory["rewards"]).float()
        masks = torch.stack(trajectory["dones"])
        act, _ = self.__actor(states)
        q_vals = self.__critic(torch.cat([states, act], dim=1))
        returns = self.__Tensor(rewards.size(0),1)
        deltas = self.__Tensor(rewards.size(0),1)
        prev_return = 0
        prev_value = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i]+self.__gamma*prev_return*masks[i]
            deltas[i] = rewards[i]+self.__gamma*prev_value*masks[i]-q_vals.data[i]
            prev_return = returns[i, 0]
            prev_value = q_vals.data[i, 0]
        deltas = (deltas-deltas.mean())/(deltas.std()+1e-10)
        mu_pi, logvar_pi = self.__actor(states)
        dist_pi = Normal(mu_pi, logvar_pi.exp().sqrt())
        pi_log_probs = dist_pi.log_prob(actions)
        ratio = (pi_log_probs-beta_log_probs.detach()).sum(dim=1, keepdim=True).exp()
        actor_loss = -torch.min(ratio*deltas, torch.clamp(ratio, 1-self.__eps, 1+self.__eps)*deltas).mean()
        cv_loss = -q_vals.mean()
        loss = actor_loss+cv_loss
        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()

class Trainer:
    def __init__(self, env_name, params):
        # initialize environment
        self.__env = gym.make(env_name)
        self.__env_name = env_name

        # save important experiment parameters for the training loop
        self.__iterations = params["iterations"]
        self.__epochs = params["epochs"]
        self.__mem_len = params["mem_len"]
        self.__seed = params["seed"]
        self.__render = params["render"]
        self.__log_interval = params["log_interval"]
        self.__warmup = params["warmup"]
        self.__batch_size = params["batch_size"]
        self.__p_batch_size = params["policy_batch_size"]
        self.__learning_updates = params["learning_updates"]
        self.__save = params["save"]

        # initialize DDPG agent using experiment parameters from config file
        state_dim = self.__env.observation_space.shape[0]
        action_dim = self.__env.action_space.shape[0]
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]
        network_settings = params["network_settings"]
        actor = Actor(state_dim, hidden_dim, action_dim)
        target_actor = Actor(state_dim, hidden_dim, action_dim)
        critic = utils.Critic(state_dim+action_dim, hidden_dim, 1)
        target_critic = utils.Critic(state_dim+action_dim, hidden_dim, 1)
        self.__agent = SCV(actor,
                        target_actor,
                        critic,
                        target_critic,
                        network_settings,
                        GPU=cuda)

        # intitialize memory
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
            batch_mean_rwd = 0
            bsize = 1
            num_episodes = 1
            s,a,lp,r,d = [],[],[],[],[]
            while bsize<= self.__p_batch_size+1:
                state = self.__Tensor(self.__env.reset())
                if ep % self.__log_interval == 0 and self.__render:
                    self.__env.render()
                    done = False
                running_reward = 0
                done = False
                t = 0
                while not done:
                    action, log_prob = self.__agent.select_action(state)
                    next_state, reward, done, _ = self.__env.step(action.data.cpu().numpy())
                    running_reward += reward
                    if ep % self.__log_interval == 0 and self.__render:
                        self.__env.render()
                    next_state = self.__Tensor(next_state)
                    reward = self.__Tensor([reward])
                    done = self.__Tensor([done])
                    self.__memory.push(state.data, action.data, next_state.data, reward, done)
                    if ep >= self.__warmup:
                        for _ in range(self.__learning_updates):
                            transitions = self.__memory.sample(self.__batch_size)
                            batch = Transition(*zip(*transitions))
                            self.__agent.online_update(self.__crit_opt, batch)
                    s.append(state)
                    a.append(action)
                    lp.append(log_prob)
                    r.append(reward)
                    d.append(done)
                    state = next_state
                    t += 1
                bsize += t
                batch_mean_rwd = (running_reward*(num_episodes-1)+running_reward)/num_episodes
                num_episodes += 1

            if (self.__best is None or running_reward > self.__best) and ep > self.__warmup and self.__save:
                self.__best = running_reward
                print("---Saving best SVG policy---")
                utils.save(self.__agent, self.__directory + "/saved_policies/ddpg-"+self.__env_name+".pth.tar")
            if ep >= self.__warmup+1:
                trajectory = {
                            "states": s,
                            "actions": a,
                            "log_probs": lp,
                            "rewards": r,
                            "dones": d
                }
                for _ in range(self.__epochs):
                    self.__agent.offline_update(self.__pol_opt, trajectory)
            interval_avg.append(batch_mean_rwd)
            avg = (avg*(ep-1)+batch_mean_rwd)/ep
            if ep % self.__log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print("Episode {}\t Interval average: {:.3f}\t Average reward: {:.3f}".format(ep, interval, avg))
                interval_avg = []
                if self.__logging:
                    self.__writer.writerow([ep, interval, avg])

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])
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

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind+flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))
    flat_grad = torch.cat(grads)
    return flat_grad
