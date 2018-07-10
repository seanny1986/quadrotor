import environments.envs as envs 
import policies.ddpg as ddpg
import argparse
import torch
import torch.nn.functional as F
import utils

class Trainer:
    def __init__(self, env_name, params):
        self.env = envs.make(env_name)

        self.iterations = params["iterations"]
        self.gamma = params["gamma"]
        self.mem_len = params["mem_len"]
        self.seed = params["seed"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.warmup = params["warmup"]
        self.batch_size = params["batch_size"]
        self.save = params["save"]
        
        action_bound = self.env.action_bound[1]
        state_dim = self.env.observation_space
        action_dim = self.env.action_space
        hidden_dim = params["hidden_dim"]
        cuda = params["cuda"]
        self.actor = ddpg.Actor(state_dim, hidden_dim, action_dim)
        self.target_actor = ddpg.Actor(state_dim, hidden_dim, action_dim)
        self.critic = ddpg.Critic(state_dim, hidden_dim, action_dim)
        self.target_critic = ddpg.Critic(state_dim, hidden_dim, action_dim)
        self.agent = ddpg.DDPG(self.actor, 
                                self.target_actor, 
                                self.critic, 
                                self.target_critic, 
                                action_bound, 
                                GPU=cuda)

        self.noise = utils.OUNoise(action_dim)
        self.noise.set_seed(self.seed)
        self.memory = ddpg.ReplayMemory(self.mem_len)

        if cuda:
            self.Tensor = torch.cuda.FloatTensor
            self.agent = self.agent.cuda()
        else:
            self.Tensor = torch.Tensor

    def train(self):
        interval_avg = []
        avg = 0
        for ep in range(1, self.iterations+1):

            state = self.Tensor(self.env.reset())
            self.noise.reset()
            running_reward = 0
            for t in range(self.env.H):
            
                # render the episode
                if ep % self.log_interval == 0:
                    self.env.render()
            
                # select an action using either random policy or trained policy
                if ep < self.warmup:
                    action = self.agent.random_action(self.noise).data
                else:
                    action = self.agent.select_action(state, noise=self.noise).data
            
                # step simulation forward
                next_state, reward, done, _ = self.env.step(action[0].cpu().numpy())
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