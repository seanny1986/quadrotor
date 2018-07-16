import environments.envs as envs 
import policies.trpo as trpo
import torch
import torch.nn.functional as F
import math
import utils
import numpy as np
import csv
import os


class Trainer:
    def __init__(self, env_name, params):
        self.env = envs.make(env_name)
        self.params = params

        self.iterations = params["iterations"]
        self.seed = params["seed"]
        self.batch_size = params["batch_size"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]
        
        cuda = params["cuda"]
        self.action_bound = self.env.action_bound[1]
        state_dim = self.env.observation_space
        action_dim = self.env.action_space
        hidden_dim = params["hidden_dim"]

        self.pi = trpo.Actor(state_dim, hidden_dim, action_dim)
        self.critic = trpo.Critic(state_dim, hidden_dim, 1)
        self.agent = trpo.TRPO(self.pi, self.critic, params["network_settings"], GPU=cuda)

        self.running_state = ZFilter((state_dim,), clip=5)
        self.running_reward = ZFilter((1,), demean=False, clip=10)

        if cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor
        
        if self.render:
            self.env.init_rendering()
        
        # initialize experiment logging
        self.logging = params["logging"]
        if self.logging:
            directory = os.getcwd()
            filename = directory + "/data/trpo.csv"
            with open(filename, "w") as csvfile:
                self.writer = csv.writer(csvfile)    
                self.train()
        else:
            self.train()
    
    def train(self):
        for i_episode in range(1, self.iterations+1):
            memory = trpo.Memory()
            num_steps = 0
            reward_batch = 0
            num_episodes = 0
            while num_steps < self.batch_size:
                state = self.env.reset()
                state = self.running_state(state[0])
                reward_sum = 0
                if self.render:
                    self.env.render()
                for t in range(1, self.iterations):
                    action = self.agent.select_action(state)
                    action = action.data[0].numpy()
                    next_state, reward, done, _ = self.env.step(action*self.action_bound)
                    reward_sum += reward
                    
                    if i_episode % self.log_interval == 0 and self.render:
                        self.env.render()

                    next_state = self.running_state(next_state[0])
                    mask = 1
                    if done:
                        mask = 0
                    memory.push(state, np.array([action]), mask, next_state, reward)
                        
                    if done:
                        break
                    state = next_state
                num_steps += (t-1)
                num_episodes += 1
                reward_batch += reward_sum
            reward_batch /= num_episodes
            batch = memory.sample()
            self.agent.update(batch)
            if i_episode % self.log_interval == 0:
                print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(i_episode, reward_sum, reward_batch))
                if self.logging:
                    self.writer.writerow([i_episode, reward]) 

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

