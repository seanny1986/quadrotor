import torch
import random
import torch.nn.functional as F
from torch.distributions import Normal

"""
    PyTorch implementation of Proximal Policy Optimization (Schulman, 2017).
"""

class PPO(torch.nn.Module):
    def __init__(self, pi, beta, critic, network_settings, GPU=True):
        super(PPO,self).__init__()
        self.__pi = pi
        self.__beta = beta
        self.__critic = critic

        self.__gamma = network_settings["gamma"]
        self.__lmbd = network_settings["lambda"]
        self.__eps = network_settings["eps"]
    
        self.__GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.__pi = self.__pi.cuda()
            self.__beta = self.__beta.cuda()
            self.__critic = self.__critic.cuda()
        else:
            self.Tensor = torch.Tensor

    def select_action(self, x):
        mu, logvar = self.__beta(x)
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
        action = torch.stack(trajectory["actions"])
        beta_log_prob = torch.stack(trajectory["log_probs"])
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

        # compute probability ratio
        mu_pi, logvar_pi = self.__pi(state)
        dist_pi = Normal(mu_pi, logvar_pi.exp().sqrt())
        pi_log_prob = dist_pi.log_prob(action)
        ratio = (pi_log_prob-beta_log_prob.detach()).sum(dim=1, keepdim=True).exp()
        optim.zero_grad()
        actor_loss = -torch.min(ratio*a_hat, torch.clamp(ratio, 1-self.__eps, 1+self.__eps)*a_hat).sum()
        critic_loss = advantage.pow(2).sum()
        loss = actor_loss+critic_loss
        self.hard_update(self.__beta, self.__pi)
        loss.backward()
        optim.step()

        

        