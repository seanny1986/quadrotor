import torch
import random
import torch.nn.functional as F
from torch.distributions import Normal

"""
    Pytorch implementation of Generalized Advantage Estimation (Schulman, 2015). Uses independent
    Gaussian actions (i.e. diagonal covariance matrix)
"""

class GAE(torch.nn.Module):
    def __init__(self, actor, critic, network_settings, GPU=True):
        super(GAE,self).__init__()
        self.__actor = actor
        self.__critic = critic

        self.__gamma = network_settings["gamma"]
        self.__lmbd = network_settings["lambda"]
    
        self.__GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.__actor = self.__actor.cuda()
            self.__critic = self.__critic.cuda()
        else:
            self.Tensor = torch.Tensor

    def select_action(self, x):
        mu, logvar = self.__actor(x)
        min_sigma = torch.ones(logvar.size())*1e-4
        if self.__GPU:
            min_sigma = min_sigma.cuda()
        std = logvar.exp().sqrt()+min_sigma
        a = Normal(mu, std)
        action = a.sample()
        logprob = a.log_prob(action)
        return F.sigmoid(action).pow(0.5), logprob

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update(self, optim, trajectory):
        state = torch.stack(trajectory["states"])
        log_prob = torch.stack(trajectory["log_probs"])
        reward = trajectory["rewards"]
        next_state = torch.stack(trajectory["next_states"])

        # compute advantage estimates
        ret = []
        gae = 0
        value = self.__critic(state)
        value_list = value.squeeze(1)
        next_value = self.__critic(next_state).squeeze(1)
        for r, v0, v1 in list(zip(reward, value_list, next_value))[::-1]:
            delta = r+self.__gamma*v1-v0
            gae = delta+self.__gamma*self.__lmbd*gae
            ret.insert(0, self.Tensor([gae]))
        ret = torch.stack(ret)
        advantage = ret-value
        a_hat = (advantage-advantage.mean())/(advantage.std()+1e-7)

        # calculate gradient and backprop
        optim.zero_grad()
        actor_loss = -(log_prob*a_hat).sum()
        critic_loss = advantage.pow(2).sum()
        loss = actor_loss+critic_loss
        loss.backward()
        optim.step()

        

        