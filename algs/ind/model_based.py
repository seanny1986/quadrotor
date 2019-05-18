import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import utils
import os
import csv

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
    
class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Policy, self).__init__()
        self.__fc1 = nn.Linear(input_dim, hidden_dim)
        self.__fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.__mu = nn.Linear(hidden_dim, output_dim)
        self.__logvar = nn.Linear(hidden_dim, output_dim)
        self.__mu.weight.data.mul_(0.1)
        self.__mu.bias.data.mul_(0.)

    def forward(self, x):
        x = F.tanh(self.__fc1(x.float()))
        x = F.tanh(self.__fc2(x))
        mu = self.__mu(x)
        logvar = self.__logvar(x)
        return mu, logvar

class ModelBasedPolicySearch(nn.Module):
    def __init__(self, policy, model, network_settings, GPU=False):
        self.model = model
        self.policy = policy

    def select_action(self, x):
        """
        Parameters
        ----------
        x :

        Returns
        -------
        action, logprob, value : tuple
            action (FloatTensor) :
                action output that is sampled from the normal distribution characterized by
                mu and logvar.
            logprob (FloatTensor) :
                log-probability of the action output characterized by mu and logvar
            value (FloatTensor) :
                critic value estimate V^{\pi}(s_{t})
        """

        mu, logvar = self.__pi(x)
        sigma = logvar.exp().sqrt()+1e-10
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def run_policy(self, env, bsize):
        while bsize<self.__batch_size+1:
            state = self.__env.reset()
            state = self.__Tensor(state)
            if ep % self.__log_interval == 0 and self.__render:
                self.__env.render()
            running_reward = 0
            done = False
            t = 0
            while not done:          
                action, log_prob = self.__agent.select_action(state)
                a = action.cpu().numpy()
                next_state, reward, done, _ = self.__env.step(a)
                running_reward += reward
                if ep % self.__log_interval == 0 and self.__render:
                    self.__env.render()
                next_state = self.__Tensor(next_state)
                reward = self.__Tensor([reward])
                state = next_state
                t += 1
            bsize += t
            batch_mean_rwd = (running_reward*(num_episodes-1)+running_reward)/num_episodes
            num_episodes += 1

    def train_policy(self):
        pass
    
    def train_model(self, optimizer, batch):
        states = torch.stack(batch["states"])
        actions = torch.stack(batch["actions"])
        next_states = torch.stack(batch["next_states"])
        state_actions = torch.cat([states, actions], dim=1)
        pred_next_states = self.model(state_actions)
        loss = F.mse_loss(pred_next_states, next_states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class Trainer:
    def __init__(self, env_name, params, ident=1):
        self.__id = str(ident)
        self.__env = gym.make(env_name)
        self.__env_name = env_name
        self.__params = params
        self.__env.set_lazy_action(True); self.__la = "la"
        self.__env.set_lazy_change(True); self.__lc = "lc"
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
        pi = Policy(state_dim, hidden_dim, action_dim)
        phi = Model(state_dim+action_dim, hidden_dim, state_dim)
        self.__agent = ModelBasedPolicySearch(pi, phi, network_settings, GPU=cuda)
        self.__optim = torch.optim.Adam(pi.parameters(), lr=learning_rate)
        self.__best = None
        if cuda:
            self.__Tensor = torch.cuda.FloatTensor
            self.__agent = self.__agent.cuda()
        else:
            self.__Tensor = torch.Tensor

        # initialize experiment logging
        self.__logging = params["logging"]
        self.__directory = os.getcwd()
        if self.__logging:
            filename = self.__directory + "/data/ppo-"+self.__id+"-"+self.__la+"-"+self.__lc+"-"+self.__env_name+".csv"
            with open(filename, "w") as csvfile:
                self.__writer = csv.writer(csvfile)
                self.__writer.writerow(["episode", "interval", "reward"])
                self._run_algo()
        else:
            self._run_algo()

    def _run_algo(self):
        interval_avg = []
        avg = 0
        for ep in range(1, self.__iterations+1):
            batch_mean_rwd = 0
            bsize = 1
            num_episodes = 1
            self.__agent.run_policy()
            self.__agent.train_model()
            self.__agent.train_policy()
            if (self.__best is None or batch_mean_rwd > self.__best) and self.__save:
                print("---Saving best model-based policy---")
                self.__best = batch_mean_rwd
                fname = self.__directory + "/saved_policies/model-based-"+self.__id+"-"+self.__la+"-"+self.__lc+"-"+self.__env_name+".pth.tar"
                utils.save(self.__agent, fname)
            interval_avg.append(batch_mean_rwd)
            avg = (avg*(ep-1)+batch_mean_rwd)/ep   
            if ep % self.__log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print('Episode {}\t Interval average: {:.3f}\t Average reward: {:.3f}'.format(ep, interval, avg))
                interval_avg = []
                if self.__logging:
                    self.__writer.writerow([ep, interval, avg])
        fname = self.__directory + "/saved_policies/ppo-"+self.__id+"-"+self.__la+"-"+self.__lc+"-"+self.__env_name+"-final.pth.tar"
        utils.save(self.__agent, fname)