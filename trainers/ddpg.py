import environments.envs as envs 
import policies.ddpg as ddpg
import argparse
import torch
import torch.nn.functional as F
import utils
import csv
import os


class Trainer:
    def __init__(self, env_name, params):
        self.env = envs.make(env_name)
        self.env_name = env_name

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
        action_bound = self.env.action_bound[1]
        state_dim = self.env.observation_space
        action_dim = self.env.action_space
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]
        network_settings = params["network_settings"]
        self.actor = ddpg.Actor(state_dim, hidden_dim, action_dim)
        self.target_actor = ddpg.Actor(state_dim, hidden_dim, action_dim)
        self.critic = ddpg.Critic(state_dim, hidden_dim, action_dim)
        self.target_critic = ddpg.Critic(state_dim, hidden_dim, action_dim)
        self.agent = ddpg.DDPG(self.actor, 
                                self.target_actor, 
                                self.critic, 
                                self.target_critic, 
                                action_bound,
                                network_settings, 
                                GPU=cuda)

        # intitialize ornstein-uhlenbeck noise for random action exploration
        ou_scale = params["ou_scale"]
        ou_mu = params["ou_mu"]
        ou_sigma = params["ou_sigma"]
        self.noise = utils.OUNoise(action_dim, scale=ou_scale, mu=ou_mu, sigma=ou_sigma)
        self.noise.set_seed(self.seed)
        self.memory = ddpg.ReplayMemory(self.mem_len)

        # send to GPU if flagged in experiment config file
        if cuda:
            self.Tensor = torch.cuda.FloatTensor
            self.agent = self.agent.cuda()
        else:
            self.Tensor = torch.Tensor
        
        if self.render:
            self.env.init_rendering()

        # initialize experiment logging
        self.logging = params["logging"]
        if self.logging:
            directory = os.getcwd()
            filename = directory + "/data/ddpg.csv"
            with open(filename, "w") as csvfile:
                self.writer = csv.writer(csvfile)
        
        self.train()

    def train(self):
        interval_avg = []
        avg = 0
        for ep in range(1, self.iterations+1):

            state = self.Tensor(self.env.reset())
            self.noise.reset()
            running_reward = 0
            if self.render:
                self.env.render()
            for t in range(self.env.H):
            
                # select an action using either random policy or trained policy
                if ep < self.warmup:
                    action = self.agent.random_action(self.noise).data
                else:
                    action = self.agent.select_action(state, noise=self.noise).data
            
                # step simulation forward
                next_state, reward, done, _ = self.env.step(action[0].cpu().numpy())
                running_reward += reward

                # render the episode
                if ep % self.log_interval == 0 and self.render:
                    self.env.render()
                
                next_state = self.Tensor(next_state)
                reward = self.Tensor([reward])

                # push to replay memory
                self.memory.push(state[0], action[0], next_state[0], reward)
            
                # online training if out of warmup phase
                if ep >= self.warmup:
                    for i in range(5):
                        transitions = self.memory.sample(self.batch_size)
                        batch = ddpg.Transition(*zip(*transitions))
                        self.agent.update(batch)

                # check if terminate
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
                    self.writer.writerow([ep, reward])
            