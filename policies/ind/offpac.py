import torch
import random
import torch.nn.functional as F
from torch.distributions import Normal

"""
    Pytorch implementation of Off-Policy Policy Gradient (Degris et al., 2012). A few modifications
    are made to the original algorithm:
    1) This implementation uses the GAE return instead of the lambda return described in Degris;
    2) I've used a neural network to parameterize the policy;
    3) I've truncated the importance weighting to only include samples *after* the current time step.
        This is not technically correct, but it performs well in practice.
"""

class OFFPAC(torch.nn.Module):
    def __init__(self, pi, beta, critic, network_settings, GPU=True):
        super(OFFPAC,self).__init__()
        self.__pi = pi
        self.__beta = beta
        self.__critic = critic

        self.__gamma = network_settings["gamma"]
        self.__lmbd = network_settings["lambda"]
        self.__lookback = network_settings["lookback"]                  # not yet implemented
    
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

    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update(self, optim, trajectory):
        state = torch.stack(trajectory["states"])
        action = torch.stack(trajectory["actions"])
        beta_log_prob = torch.stack(trajectory["log_probs"])
        reward = trajectory["rewards"]
        next_state = torch.stack(trajectory["next_states"])

        # compute advantage estimates and importance sampling correction
        ret = []
        wts = []
        log_probs = []
        gae = 0
        w = 0
        value = self.__critic(state)
        next_value = self.__critic(next_state).detach()
        for s, a, r, v0, v1, blp in list(zip(state, action, reward, value, next_value, beta_log_prob))[::-1]:
            pi_mu, pi_logvar = self.__pi(s.unsqueeze(0))
            pi_dist = Normal(pi_mu, pi_logvar.exp().sqrt())
            pi_log_prob = pi_dist.log_prob(a)
            ratio = pi_log_prob.detach()-blp
            w += ratio
            log_probs.insert(0, pi_log_prob)
            wts.insert(0, w)
            delta = r+self.__gamma*v1-v0
            gae = delta+self.__gamma*self.__lmbd*gae
            ret.insert(0, self.Tensor([gae]))
        ret = torch.stack(ret)
        wts = torch.stack(wts)
        log_probs = torch.stack(log_probs)
        advantage = ret-value
        a_hat = (advantage-advantage.mean())/advantage.std()

        # set current pi as new beta
        self._hard_update(self.__beta, self.__pi)

        # calculate gradient and backprop
        optim.zero_grad()
        actor_loss = -(wts.exp()*log_probs*a_hat).sum()
        critic_loss = advantage.pow(2).sum()
        loss = actor_loss+critic_loss
        loss.backward()
        optim.step()

        

        