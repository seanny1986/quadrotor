import torch
import random
import torch.nn.functional as F
from torch.distributions import Normal

"""
    Pytorch implementation of Generalized Advantage Estimation (Schulman, 2015). Uses independent
    Gaussian actions (i.e. diagonal covariance matrix)
"""

class GAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, network_settings, GPU=True):
        super(GAE,self).__init__()
        self.__l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.__mu = torch.nn.Linear(hidden_dim, output_dim)
        self.__logvar = torch.nn.Linear(hidden_dim, output_dim)
        self.__value = torch.nn.Linear(hidden_dim, 1)

        self.__gamma = network_settings["gamma"]
        self.__lmbd = network_settings["lambda"]
    
        self.__GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def forward(self, x):
        x = F.tanh(self.__l1(x))
        mu = self.__mu(x)
        logvar = self.__logvar(x)
        value = self.__value(x)
        return mu, logvar, value

    def select_action(self, x):
        mu, logvar, value = self.forward(x)     
        min_sigma = torch.ones(logvar.size())*1e-4
        if self.__GPU:
            min_sigma = min_sigma.cuda()
        std = logvar.exp().sqrt()+min_sigma
        a = Normal(mu, std)
        action = a.sample()
        logprob = a.log_prob(action)
        return F.sigmoid(action).pow(0.5), logprob, value

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update(self, optim, trajectory):
        log_probs = torch.stack(trajectory["log_probs"]).float()
        rewards = torch.stack(trajectory["rewards"]).float()
        values = torch.stack(trajectory["values"]).float()
        masks = torch.stack(trajectory["dones"])

        # compute advantage estimates
        returns = self.Tensor(rewards.size(0),1)
        deltas = self.Tensor(rewards.size(0),1)
        advantages = self.Tensor(rewards.size(0),1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i]+self.__gamma*prev_return*masks[i]
            deltas[i] = rewards[i]+self.__gamma*prev_value*masks[i]-values.data[i]
            advantages[i] = deltas[i]+self.__gamma*self.__lmbd*prev_advantage*masks[i]
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
        
        advantages = returns-values
        a_hat = (advantages-advantages.mean())/(advantages.std()+1e-7)

        # calculate gradient and backprop
        optim.zero_grad()
        actor_loss = -(log_probs*a_hat).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss+critic_loss
        loss.backward(retain_graph=True)
        optim.step()

        

        