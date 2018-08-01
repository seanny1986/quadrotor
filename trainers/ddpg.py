import environments.envs as envs 
import policies.ind.ddpg as ddpg
import argparse
import torch
import torch.nn.functional as F
import utils
import csv
import os


class Trainer:
    def __init__(self, env_name, params):
        # initialize environment
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
        self.action_bound = self.env.action_bound[1]
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]
        network_settings = params["network_settings"]
        actor = ddpg.Actor(state_dim, hidden_dim, action_dim)
        target_actor = ddpg.Actor(state_dim, hidden_dim, action_dim)
        critic = utils.Critic(state_dim+action_dim, hidden_dim, 1)
        target_critic = utils.Critic(state_dim+action_dim, hidden_dim, 1)
        self.agent = ddpg.DDPG(actor, 
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
        self.memory = utils.ReplayMemory(self.mem_len)

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
                next_state, reward, done, info = self.env.step(action[0].cpu().numpy()*self.action_bound)
                running_reward += reward

                # render the episode if render selected
                if ep % self.log_interval == 0 and self.render:
                    self.env.render()
                
                # transform to tensors before storing in memory
                next_state = self.Tensor(next_state)
                reward = self.Tensor([reward])

                # push to replay memory
                self.memory.push(state[0], action[0], next_state[0], reward)
            
                # online training if out of warmup phase
                if ep >= self.warmup:
                    for i in range(5):
                        transitions = self.memory.sample(self.batch_size)
                        batch = utils.Transition(*zip(*transitions))
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
                print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
                interval_avg = []
                if self.logging:
                    self.writer.writerow([ep, avg])
            