import environments.envs as envs 
import policies.ind.fmis as fmis
import argparse
import torch
import torch.nn.functional as F
import utils
import csv
import os
import numpy as np


class Trainer:
    def __init__(self, env_name, params):
        self.env = envs.make(env_name)
        self.params = params
        self.action_bound = self.env.action_bound[1]

        self.iterations = params["iterations"]
        self.seed = params["seed"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]
        
        self.cuda = params["cuda"]
        state_dim = self.env.observation_space
        action_dim = self.env.action_space
        hidden_dim = params["hidden_dim"]
        network_settings = params["network_settings"]

        pi = utils.Actor(state_dim, hidden_dim, action_dim)
        beta = utils.Actor(state_dim, hidden_dim, action_dim)
        critic = utils.Critic(state_dim, hidden_dim, 1)
        self.agent = fmis.FMIS(pi, beta, critic, self.env, network_settings, GPU=self.cuda)

        self.pi_optim = torch.optim.Adam(self.agent.parameters())

        self.memory = fmis.ReplayMemory(1000000)

        if self.cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

        self.best = None
        
        # use OU noise to explore and learn the model for n warmup episodes
        self.noise = utils.OUNoise(action_dim, mu=10)
        self.warmup = 5

        # initialize experiment logging
        self.logging = params["logging"]
        if self.logging:
            self.directory = os.getcwd()
            filename = self.directory + "/data/fmis.csv"
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
            running_reward = 0
            state = self.Tensor(self.env.reset())
            s0 = state.clone()
            if self.render:
                self.env.render()
            for _ in range(self.env.H):
                if ep<self.warmup+1:
                    a = np.array([self.noise.noise()], dtype="float32")
                    action = torch.from_numpy(a)
                    if self.cuda:
                        action = action.cuda()
                else:
                    action, _ = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action[0].cpu().numpy()*self.action_bound)
                running_reward += reward
                next_state = self.Tensor(next_state)
                
                # push to replay memory
                self.memory.push(state[0], action[0], next_state[0])

                if ep % self.log_interval == 0 and self.render:
                    self.env.render()
                
                if done:
                    break
                state = next_state
            
            if (self.best is None or running_reward > self.best) and self.save:
                self.best = running_reward
                utils.save(self.agent, self.directory + "/saved_policies/fmis.pth.tar")
            
            self.agent.model_update(self.memory)
            if ep > self.warmup+1:
                print("---OPTIMIZING POLICY---")
                for _ in range(100):
                    self.agent.policy_update(self.pi_optim, s0, self.env.H)
                
            interval_avg.append(running_reward)
            avg = (avg*(ep-1)+running_reward)/ep
            if ep % self.log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print("Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}".format(ep, interval, avg))
                interval_avg = []
                if self.logging:
                    self.writer.writerow([ep, avg])  