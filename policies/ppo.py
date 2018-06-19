import torch
import random
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, output_dim)
        self.logvar = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar 

class Critic(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.v = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        value = self.v(x)
        return value

class PPO(torch.nn.Module):
    def __init__(self, actor, critic, target_actor, gamma=0.99, lmbd=0.92, eps=0.02):
        super(PPO,self).__init__()
        self.actor = actor
        self.critic = critic
        self.pi_old = target_actor

        self.gamma = gamma
        self.lmbd = lmbd
        self.eps = eps
    
    def select_action(self, x):
        mu, logvar = self.actor(x)
        a = Normal(mu, logvar.exp().sqrt())
        action = a.sample()
        log_prob = a.log_prob(action)
        return action, log_prob

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update(self, optim, trajectory):
        state = torch.stack(trajectory["states"])
        action = torch.stack(trajectory["actions"])
        log_prob = torch.stack(trajectory["log_probs"])
        reward = trajectory["rewards"]
        next_state = torch.stack(trajectory["next_states"])

        # compute advantage estimates
        ret = []
        gae = 0
        value = self.critic(state)
        value_list = value.squeeze(1).tolist()
        next_value = self.critic(next_state).squeeze(1).tolist()
        for r, v0, v1 in list(zip(reward, value_list, next_value))[::-1]:
            delta = r+self.gamma*v1-v0
            gae = delta+self.gamma*self.lmbd*gae
            ret.insert(0, torch.Tensor([gae]))
        ret = torch.stack(ret)
        advantage = ret-value
        a_hat = (advantage-advantage.mean())/advantage.std()

        # compute probability ratio
        mu_old, logvar_old = self.pi_old(state)
        dist_old = Normal(mu_old, logvar_old.exp().sqrt())
        ratio = (log_prob-dist_old.log_prob(action)).sum(dim=1, keepdim=True).exp()
        optim.zero_grad()
        actor_loss = -torch.min(ratio*a_hat, torch.clamp(ratio, 1-self.eps, 1+self.eps)*a_hat).sum()
        critic_loss = advantage.pow(2).sum()
        loss = actor_loss+critic_loss
        self.hard_update(self.pi_old, self.actor)
        loss.backward()
        optim.step()

        

        