import environments.envs as envs 
import policies.ind.gae as gae
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
        self.agent = gae.GAE(state_dim, hidden_dim, action_dim, network_settings, GPU=cuda)
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
            filename = self.directory + "/data/gae.csv"
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
            
            s_ = []
            a_ = []
            ns_ = []
            r_ = []
            v_ = []
            lp_ = []
            dones = []
            batch_mean_rwd = 0
            for i in range(1, self.batch_size+1):
                state = self.Tensor(self.env.reset())
                if ep % self.log_interval == 0 and self.render:
                    self.env.render()
                running_reward = 0
                for _ in range(self.env.H):          
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
            
                if (self.best is None or running_reward > self.best) and self.save:
                    self.best = running_reward
                    utils.save(self.agent, self.directory + "/saved_policies/gae.pth.tar")
                batch_mean_rwd = (running_reward*(i-1)+running_reward)/i
            
            trajectory = {"states": s_,
                        "actions": a_,
                        "next_states": ns_,
                        "rewards": r_,
                        "dones": dones,
                        "values": v_,
                        "log_probs": lp_}
            for i in range(self.epochs):
                self.agent.update(self.optim, trajectory)
            interval_avg.append(batch_mean_rwd)
            avg = (avg*(ep-1)+running_reward)/ep   
            if ep % self.log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
                interval_avg = []
                if self.logging:
                    self.writer.writerow([ep, avg])