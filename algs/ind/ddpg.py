import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import gym
import gym_aero
import utils
import csv
import os
import numpy as np
from collections import namedtuple
import random

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.__affine1 = nn.Linear(state_dim, hidden_dim)
        self.__action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.tanh(self.__affine1(x))
        mu = self.__action_head(x)
        return mu

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
            sigma = self.Tensor(noise.noise())
            return F.tanh(mu+sigma)
        else:
            return F.tanh(mu)

    def random_action(self, noise):
        action = self.Tensor(noise.noise())
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
            done = Variable(torch.cat(batch.done))
        reward = torch.unsqueeze(reward, 1)
        done = torch.unsqueeze(done, 1)

        next_action = self.__target_actor(next_state)                                                 # take off-policy action
        next_state_action = torch.cat([next_state, next_action],dim=1)                              # next state-action batch
        next_state_action_value = self.__target_critic(next_state_action)                             # target q-value
        with torch.no_grad():
            expected_state_action_value = reward+self.__gamma*next_state_action_value*(1-done)             # value iteration

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


class Trainer:
    def __init__(self, env_name, params):
        # initialize environment
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.trim = np.array(self.env.trim)

        # save important experiment parameters for the training loop
        self.iterations = params["iterations"]
        self.mem_len = params["mem_len"]
        self.seed = params["seed"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.warmup = params["warmup"]
        self.batch_size = params["batch_size"]
        self.save = params["save"]
        
        # initialize DDPG agent using experiment parameters from config file
        self.action_bound = self.env.action_bound[1]
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]
        network_settings = params["network_settings"]
        actor = Actor(state_dim, hidden_dim, action_dim)
        target_actor = Actor(state_dim, hidden_dim, action_dim)
        critic = utils.Critic(state_dim+action_dim, hidden_dim, 1)
        target_critic = utils.Critic(state_dim+action_dim, hidden_dim, 1)
        self.agent = DDPG(actor, 
                        target_actor, 
                        critic, 
                        target_critic,
                        network_settings, 
                        GPU=cuda)

        # intitialize ornstein-uhlenbeck noise for random action exploration
        ou_scale = params["ou_scale"]
        ou_mu = params["ou_mu"]
        ou_sigma = params["ou_sigma"]
        self.noise = utils.OUNoise(action_dim, scale=ou_scale, mu=ou_mu, sigma=ou_sigma)
        self.noise.set_seed(self.seed)
        self.memory = ReplayMemory(self.mem_len)

        self.pol_opt = torch.optim.Adam(actor.parameters())
        self.crit_opt = torch.optim.Adam(critic.parameters())

        # want to save the best policy
        self.best = None

        # send to GPU if flagged in experiment config file
        if cuda:
            self.Tensor = torch.cuda.FloatTensor
            self.agent = self.agent.cuda()
        else:
            self.Tensor = torch.Tensor

        # initialize experiment logging. This wipes any previous file with the same name
        self.logging = params["logging"]
        if self.logging:
            self.directory = os.getcwd()
            filename = self.directory + "/data/ddpg.csv"
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

            state = self.Tensor(self.env.reset())
            self.noise.reset()
            running_reward = 0
            if ep % self.log_interval == 0 and self.render:
                self.env.render()
            for t in range(self.env.H):
            
                # select an action using either random policy or trained policy
                if ep < self.warmup:
                    action = self.agent.random_action(self.noise).data
                else:
                    action = self.agent.select_action(state, noise=self.noise).data

                # step simulation forward
                a = self.trim+action.cpu().numpy()*15
                next_state, reward, done, _ = self.env.step(a)
                running_reward += reward

                # render the episode if render selected
                if ep % self.log_interval == 0 and self.render:
                    self.env.render()
                
                # transform to tensors before storing in memory
                next_state = self.Tensor(next_state)
                reward = self.Tensor([reward])
                done = self.Tensor([done])

                # push to replay memory
                self.memory.push(state, action, next_state, reward, done)
            
                # online training if out of warmup phase
                if ep >= self.warmup:
                    for i in range(3):
                        transitions = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        self.agent.update(batch, self.crit_opt, self.pol_opt)

                # check if terminate
                if done:
                    break

                # step to next state
                state = next_state

            if (self.best is None or running_reward > self.best) and self.save:
                self.best = running_reward
                print("Saving best DDPG model.")
                utils.save(self.agent, self.directory + "/saved_policies/ddpg.pth.tar")

            # anneal noise 
            if ep > self.warmup:
                self.noise.anneal()

            # print running average and interval average, log average to csv file
            interval_avg.append(running_reward)
            avg = (avg*(ep-1)+running_reward)/ep   
            if ep % self.log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print("Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}".format(ep, interval, avg))
                interval_avg = []
                if self.logging:
                    self.writer.writerow([ep, avg])


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
            