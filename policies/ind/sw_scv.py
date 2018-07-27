import torch
import random
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple

class Sleepwalk(torch.nn.Module):
    def __init__(self, actor, critic, target_actor, target_critic, network_settings, GPU=False):
        super(Sleepwalk,self).__init__()
        self.__actor = actor
        self.__critic = critic
        self.__target_actor = target_actor
        self.__target_critic = target_critic

        self.__crit_loss = torch.nn.L1Loss()

        self.__gamma = network_settings["gamma"]
        self.__tau = network_settings["tau"]

        self._hard_update(target_actor, actor)
        self._hard_update(target_critic, critic)

        self.__GPU = GPU
        
        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.__actor = self.__actor.cuda()
            self.__target_actor = self.__target_actor.cuda()
            self.__critic = self.__critic.cuda()
            self.__target_critic = self.__target_critic.cuda()
        else:
            self.Tensor = torch.Tensor

    def _soft_update(self, target, source, tau):
	    for target_param, param in zip(target.parameters(), source.parameters()):
		    target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)
    
    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
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
    
    def online_update(self, optim, batch):
        state = torch.stack(batch.state)
        action = torch.stack(batch.action)
        reward = torch.stack(batch.reward)
        next_state = torch.stack(batch.next_state)

        # update state-action value function off-policy
        with torch.no_grad():
            next_action_mu, _  = self.__target_actor(next_state)
            q_next = self.__target_critic(torch.cat([next_state, next_action_mu], dim=1))
        target = reward+self.__gamma*q_next
        q = self.__critic(torch.cat([state, action],dim=1))
        q_loss = self.__crit_loss(q, target)
        optim.zero_grad()
        q_loss.backward()
        optim.step()

        # soft update of critic (polyak averaging)
        self._soft_update(self.__target_critic, self.__critic, self.__tau)

    def offline_update(self, optim, trajectory):
        state = trajectory["states"]
        action = trajectory["actions"]
        action_logprobs = trajectory["log_probs"]
        reward = trajectory["rewards"]
        rewards = []
        R = 0.
        for r in reward[::-1]:
            R = r.float()+self.__gamma*R
            rewards.insert(0, R)
        q_act = torch.stack(rewards)
        state = torch.stack(state).squeeze(1)
        action = torch.stack(action).squeeze(1)
        action_logprobs = torch.stack(action_logprobs).squeeze(1)

        q_vals = self.__critic(torch.cat([state, action],dim=1))
        q_hat = q_act-q_vals.detach()
        q_hat = (q_hat-q_hat.mean())/(q_hat.std()+1e-7)
        optim.zero_grad()
        loss = -(action_logprobs.sum(dim=1, keepdim=True)*(q_hat+q_vals)).sum()
        loss.backward()
        optim.step()
        self._soft_update(self.__target_actor, self.__actor, self.__tau)