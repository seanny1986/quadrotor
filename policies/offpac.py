import torch
import random
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

"""
    Pytorch implementation of Off-Policy Policy Gradient (Degris et al., 2012). A few modifications
    are made to the original algorithm:
    1) This implementation uses the GAE return instead of the lambda return described in Degris;
    2) I've used a neural network to parameterize the policy;
    3) I've truncated the importance weighting to only include samples *after* the current time step.
        This is not technically correct, but it performs well in practice.
"""

class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, output_dim)
        self.non_diag = torch.nn.Linear(hidden_dim, int(output_dim*(output_dim+1)/2-output_dim))
        self.diag = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        mu = self.mu(x)
        non_diag = self.non_diag(x)
        diag = F.softplus(self.diag(x))
        diag = diag+torch.ones(diag.size())*1e-3                                # min sigma value to prevent matrix degeneracy
        A = torch.zeros(x.size()[0], self.output_dim, self.output_dim)
        for i in range(self.output_dim):
            A[:,i,i] = diag[:,i]
        A[:,1:,0] = non_diag[:,0:3]
        A[:,2:,1] = non_diag[:,3:5]
        A[:,3:,2] = non_diag[:,5:]
        return mu, A

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

class OFFPAC(torch.nn.Module):
    def __init__(self, pi, beta, critic, network_settings, GPU=True):
        super(OFFPAC,self).__init__()
        self.pi = pi
        self.beta = beta
        self.critic = critic

        self.gamma = network_settings["gamma"]
        self.lmbd = network_settings["lambda"]
        self.lookback = network_settings["lookback"]
    
        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()
        else:
            self.Tensor = torch.Tensor

    def select_action(self, x):
        mu, A = self.beta(x)
        a = MultivariateNormal(mu, scale_tril=A)
        action = a.sample()
        log_prob = a.log_prob(action)
        return F.sigmoid(action).pow(0.5), log_prob

    def hard_update(self, target, source):
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
        value = self.critic(state)
        next_value = self.critic(next_state).detach()
        for s, a, r, v0, v1, blp in list(zip(state, action, reward, value, next_value, beta_log_prob))[::-1]:
            pi_mu, pi_A = self.pi(s.unsqueeze(0))
            pi_dist = MultivariateNormal(pi_mu, scale_tril=pi_A)
            pi_log_prob = pi_dist.log_prob(a)
            ratio = pi_log_prob.detach()-blp
            w += ratio
            log_probs.insert(0, pi_log_prob)
            wts.insert(0, w)
            delta = r+self.gamma*v1-v0
            gae = delta+self.gamma*self.lmbd*gae
            ret.insert(0, self.Tensor([gae]))
        ret = torch.stack(ret)
        wts = torch.stack(wts)
        log_probs = torch.stack(log_probs)
        advantage = ret-value
        a_hat = (advantage-advantage.mean())/advantage.std()

        # set current pi as new beta
        self.hard_update(self.beta, self.pi)

        # calculate gradient and backprop
        optim.zero_grad()
        actor_loss = -(wts.exp()*log_probs*a_hat).sum()
        critic_loss = advantage.pow(2).sum()
        loss = actor_loss+critic_loss
        loss.backward()
        optim.step()

        

        