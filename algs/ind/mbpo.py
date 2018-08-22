import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple
import os
import csv
import gym
import gym_aero

class Dynamics(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Dynamics,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return self.fc2(x)

class DeterministicPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeterministicPolicy,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return self.fc2(x)

class PolicySearch(nn.Module):
    def __init__(self, policy, dynamics, params):
        super(PolicySearch,self).__init__()
        self.policy = policy
        self.dynamics = dynamics
        self.dyn_opt = torch.optim.Adam(self.dynamics.Parameters())
        self.thresh = 0.1
    
    def select_action(self, state):
        return self.policy(state)

    def update_model(self, batch):
        states = torch.stack(batch["states"])
        actions = torch.stack(batch["actions"])
        next_states = torch.stack(batch["next_states"])
        state_actions = torch.cat([states, actions], dim=1)
        deltas = next_states-states
        pred_deltas = self.dynamics(state_actions)
        loss = F.mse_loss(pred_deltas, deltas)
        self.dyn_opt.zero_grad()
        loss.backward()
        self.dyn_opt.step()

    def update_policy(self, optim, states):
        targets = states[1:,:]

        s, idx = random.sample(states)
        done = False
        while not done:
            t = targets[idx]
            s_t = torch.cat([s, t], dim=1)
            a = self.select_action(s_t)
            s_a = torch.cat([s, a], dim=1)
            n_s = self.dynamics(s_a)
            loss = F.mse_loss(n_s, t)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if idx == len(states): break
            idx += 1

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
        dynamics = Dynamics(state_dim+action_dim, hidden_dim, state_dim)
        policy = DeterministicPolicy(state_dim+3, hidden_dim, action_dim)
        self.agent = PolicySearch(policy, dynamics, network_settings)
        self.__optim = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.__best = None
        self.thresh = 0.1
        self.memory = ReplayMemory(1000000)
        self.set_trajectory()
        self.targets = self.trajectory[1:]
        if cuda:
            self.__Tensor = torch.cuda.FloatTensor
            self.__agent = self.__agent.cuda()
        else:
            self.__Tensor = torch.Tensor

        # initialize experiment logging
        self.__logging = params["logging"]
        self.__directory = os.getcwd()
        if self.__logging:
            filename = self.__directory + "/data/mbpo-"+self.__env_name+".csv"
            with open(filename, "w") as csvfile:
                self.__writer = csv.writer(csvfile)
                self.__writer.writerow(["episode", "reward"])
                self._run_algo()
        else:
            self._run_algo()
        
    def set_trajectory(self, traj=None):
        if traj is not None:
            self.trajectory = traj
        else:
            xs = np.array([0.05*x for x in range(40)])
            ys = np.array([0 for _ in range(40)])
            zs = np.array([0 for _ in range(40)])
            self.trajectory = np.vstack([xs, ys, zs]).T

    def _run_algo(self):
        idx = random.randint(0, len(self.trajectory)-1)
        xyz = self.trajectory[idx]
        done = False
        while not done:
            target = self.targets[idx]
            state_target = torch.cat([state, target], dim=1)
            action = self.agent.select_action(state_target)
            next_state, _, done, _ = self.env.step(action)
            self.memory.push([state, action, next_state])
            if np.linalg.norm(next_state-target) > self.thresh: break
            if idx == len(self.states): break
            idx += 1
        for _ in range(self.dynamics_epochs):
            batch = self.memory.sample(self.batch_size)
            self.agent.update_model(batch)
        for _ in range(self.policy_epochs):
            self.agent.update_policy(self.trajectory)

Transition = namedtuple("Transition", ["state", "action", "next_state"])
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




