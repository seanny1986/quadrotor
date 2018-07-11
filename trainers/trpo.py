import environments.envs as envs 
import policies.trpo as trpo
import argparse
import torch
import torch.nn.functional as F
import math
import utils
import numpy as np

class Trainer:
    def __init__(self, env_name, params):
        self.env = envs.make(env_name)
        self.params = params

        self.iterations = params["iterations"]
        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.l2_reg = params["l2_reg"]
        self.max_kl = params["max_kl"]
        self.damping = params["damping"]
        self.seed = params["seed"]
        self.batch_size = params["batch_size"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]
        
        cuda = params["cuda"]
        state_dim = self.env.observation_space
        action_dim = self.env.action_space
        hidden_dim = params["hidden_dim"]

        self.pi = trpo.Actor(state_dim, hidden_dim, action_dim)
        self.critic = trpo.Critic(state_dim, hidden_dim, 1)
        self.agent = trpo.TRPO(self.pi, self.beta, self.critic, self.phi, self.env, GPU=cuda)

        self.pi_optim = torch.optim.Adam(self.pi.parameters())
        self.phi_optim = torch.optim.Adam(self.phi.parameters())

        if cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor
        
        if self.render:
            self.env.init_rendering()
            
        self.train()
    
    def train(self):
        for i_episode in range(1, self.iterations+1):
            memory = Memory()
            num_steps = 0
            reward_batch = 0
            num_episodes = 0
            while num_steps < self.batch_size:
                state = self.env.reset()
                state = running_state(state)
                reward_sum = 0
                for t in range(10000): # Don't infinite loop while learning
                    action = select_action(state)
                    action = action.data[0].numpy()
                    next_state, reward, done, _ = env.step(action)
                    reward_sum += reward
                    next_state = running_state(next_state)
                    mask = 1
                    if done:
                        mask = 0
                    memory.push(state, np.array([action]), mask, next_state, reward)

                    if i_episode % self.log_interval == 0 and self.render:
                        self.env.render()
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

