import torch
import random
import torch.nn.functional as F
from torch.distributions import Normal

"""
    PyTorch implementation of Proximal Policy Optimization (Schulman, 2017).
"""

class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, GPU=True):
        super(Actor,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, output_dim)
        self.logvar = torch.nn.Linear(hidden_dim, output_dim)

        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def forward(self, x):
        x = F.tanh(self.l1(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar 

class Critic(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, GPU=True):
        super(Critic,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.v = torch.nn.Linear(hidden_dim, output_dim)

        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def forward(self, x):
        x = F.tanh(self.l1(x))
        value = self.v(x)
        return value

class PPO(torch.nn.Module):
    def __init__(self, pi, critic, beta, env, gamma=0.99, lmbd=0.92, eps=0.2, GPU=True):
        super(PPO,self).__init__()
        self.pi = pi
        self.critic = critic
        self.beta = beta
        self.env = env
        self.action_bound = env.action_bound

        self.gamma = gamma
        self.lmbd = lmbd
        self.eps = eps
    
        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.pi = self.pi.cuda()
            self.critic = self.critic.cuda()
            self.beta = self.beta.cuda()
        else:
            self.Tensor = torch.Tensor

    def select_action(self, x):
        mu, logvar = self.beta(x)
        a = Normal(mu, logvar.exp().sqrt())
        action = a.sample()
        log_prob = a.log_prob(action)
        return F.sigmoid(action)*self.action_bound[1], log_prob

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
        value = self.critic(state)
        value_list = value.squeeze(1).tolist()
        next_value = self.critic(next_state).squeeze(1).tolist()
        for r, v0, v1 in list(zip(reward, value_list, next_value))[::-1]:
            delta = r+self.gamma*v1-v0
            gae = delta+self.gamma*self.lmbd*gae
            ret.insert(0, self.Tensor([gae]))
        ret = torch.stack(ret)
        advantage = ret-value
        a_hat = (advantage-advantage.mean())/advantage.std()

        # compute probability ratio
        mu_pi, logvar_pi = self.pi(state)
        dist_pi = Normal(mu_pi, logvar_pi.exp().sqrt())
        pi_log_prob = dist_pi.log_prob(action)
        ratio = (pi_log_prob-beta_log_prob).sum(dim=1, keepdim=True).exp()
        optim.zero_grad()
        actor_loss = -torch.min(ratio*a_hat, torch.clamp(ratio, 1-self.eps, 1+self.eps)*a_hat).sum()
        critic_loss = advantage.pow(2).sum()
        loss = actor_loss+critic_loss
        self.hard_update(self.beta, self.pi)
        loss.backward()
        optim.step()

        

        