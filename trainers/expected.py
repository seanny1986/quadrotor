import environments.envs as envs 
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
        self.trim = np.array(self.env.trim)
        self.action_delta = 50

        self.iterations = params["iterations"]
        self.seed = params["seed"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]
        
        # initialize experiment logging
        self.logging = params["logging"]
        if self.logging:
            directory = os.getcwd()
            filename = directory + "/data/expected.csv"
            with open(filename, "w") as csvfile:
                self.writer = csv.writer(csvfile)
                self.writer.writerow(["episode", "reward"])
                self.train()
        else:
            self.train()

    def select_action(self):
        return self.trim+np.random.randn(4)*self.action_delta

    def train(self):
        interval_avg = []
        avg = 0
        for ep in range(1, self.iterations+1):
            running_reward = 0
            self.env.reset()
            if ep % self.log_interval == 0 and self.render:
                self.env.render()
            for _ in range(self.env.H):          
                action = self.select_action()
                next_state, reward, done, info = self.env.step(action)
                running_reward += reward
                if ep % self.log_interval == 0 and self.render:
                    self.env.render()
                if done:
                    break
                state = next_state
            interval_avg.append(running_reward)
            avg = (avg*(ep-1)+running_reward)/ep   
            if ep % self.log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
                interval_avg = []
                if self.logging:
                    self.writer.writerow([ep, avg])