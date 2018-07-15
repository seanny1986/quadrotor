import torch
import random
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple

"""
    Implements an architecture I'm calling Forward Model Importance Sampling. A statistical RNN 
    forward model is trained using the log-likelihood score function. The agent is trained under
    this model using PPO. 
"""

class Dynamics(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Dynamics,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = torch.nn.LSTMCell(input_dim, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, output_dim)
        self.logvar = torch.nn.Linear(hidden_dim, output_dim)

        self.GPU = True

    def set_cuda(self, GPU):
        self.GPU = GPU

    def forward(self, s0, H):
        xs = []
        lps = []
        hs = []
        h_t = torch.zeros(1, self.hidden_dim, dtype=torch.float)
        c_t = torch.zeros(1, self.hidden_dim, dtype=torch.float)
        if self.GPU:
                h_t = h_t.cuda()
                c_t = c_t.cuda()
        for t in range(H):
            h_t, c_t = self.lstm(s0[t,:].unsqueeze(0), (h_t, c_t))
            mu = self.mu(h_t)
            logvar = self.logvar(h_t)
            dist = Normal(mu, logvar.exp().sqrt())
            x = dist.sample()
            lps.append(dist.log_prob(x))
            xs.append(x)
            hs.append(h_t)
        return xs, lps, hs
    
    def step(self, state_action, args=None, sample=False):
        if args is None:
            c_t = torch.zeros(1, self.hidden_dim, dtype=torch.float)
            h_t = torch.zeros(1, self.hidden_dim, dtype=torch.float)
            if self.GPU:
                h_t = h_t.cuda()
                c_t = c_t.cuda()
            args = (h_t, c_t)
        h_t, c_t = self.lstm(state_action, args)
        mu = self.mu(h_t)
        if not sample:
            return mu, (h_t, c_t)
        else:
            logvar = self.logvar(h_t)
            dist = Normal(mu, logvar.exp().sqrt())
            x = dist.sample()
            return x, (h_t, c_t)
    
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

class FMIS(torch.nn.Module):
    def __init__(self, pi, beta, critic, phi, env, gamma=0.99, eps=0.2, lmbd=0.92, GPU=True):
        super(FMIS,self).__init__()
        self.pi = pi
        self.critic = critic
        self.beta = beta
        self.phi = phi
        self.gamma = gamma
        self.eps = eps
        self.lmbd = lmbd
        self.GPU = GPU

        self.s0_mu = None
        self.s0_logvar = None
        self.env = env
        self.observation_space = env.observation_space
        self.action_bound = env.action_bound[1]

        self.state = []
        self.action = []
        self.log_prob = []
        self.next_state = []
        self.reward = []

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.pi = self.pi.cuda()
            self.critic = self.critic.cuda()
            self.beta = self.beta.cuda()
            self.phi = phi.cuda()
            self.phi.set_cuda(GPU)
        else:
            self.Tensor = torch.Tensor

    def rollout(self, s0, H):
        # rollout trajectory under model
        x = s0
        args = None
        for t in range(H):
            self.state.append(x)
            mu, logvar = self.beta(x)
            dist = Normal(mu, logvar.exp().sqrt())
            action = dist.sample()
            beta_logprob = dist.log_prob(action)
            state_action = torch.cat([x, action],dim=1)
            x, args = self.phi.step(state_action, args)
            xyz = x[:,0:3].detach().cpu().numpy().T
            a = action.squeeze(0).cpu().numpy()
            r = self.Tensor([self.env.reward(xyz, a)])
            self.next_state.append(x[:,0:self.observation_space])
            self.action.append(action)
            self.log_prob.append(beta_logprob)
            self.reward.append(r)
        observations = {"states": self.state,
                        "actions": self.action,
                        "log_probs": self.log_prob,
                        "next_states": self.next_state,
                        "rewards": self.reward}
        return observations

    def select_action(self, x):
        mu, logvar = self.beta(x)
        sigma = logvar.exp().sqrt()+torch.ones(x.size()[0], self.actor.output_dim)*1e-4
        a = Normal(mu, sigma)
        action = a.sample()
        log_prob = a.log_prob(action)
        return F.sigmoid(action)*self.action_bound, log_prob

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def model_update(self, optim, trajectory):
        state = torch.stack(trajectory["states"])
        action = torch.stack(trajectory["actions"])
        next_state = torch.stack(trajectory["next_states"])
        
        # compute model loss
        H = len(action)
        xs = torch.cat([state, action],dim=1)
        ys = next_state
        pred_ys, log_probs, _ = self.phi(xs, H)
        pred_ys, log_probs = torch.stack(pred_ys).squeeze(1), torch.stack(log_probs).squeeze(1)
        error = (pred_ys-ys)**2
        model_loss = (log_probs*error).mean()
        optim.zero_grad()
        model_loss.backward()
        optim.step()
        return error.sum().item()

    def policy_update(self, optim, s0, H):

        # rollout trajectory under statistical model
        trajectory = self.rollout(s0.detach(), H)
        state = torch.stack(trajectory["states"]).squeeze(1).detach()
        action = torch.stack(trajectory["actions"]).squeeze(1).detach()
        next_state = torch.stack(trajectory["next_states"]).squeeze(1).detach()
        reward = torch.stack(trajectory["rewards"]).squeeze(1).detach()
        beta_logprob = torch.stack(trajectory["log_probs"]).squeeze(1).detach()

        # compute advantage estimates
        ret = []
        gae = 0
        value = self.critic(state).squeeze(1)
        next_value = self.critic(next_state).squeeze(1)
        for r, v0, v1 in list(zip(reward, value, next_value))[::-1]:
            delta = r+self.gamma*v1-v0
            gae = delta+self.gamma*self.lmbd*gae
            ret.insert(0, self.Tensor([gae]))
        ret = torch.stack(ret)
        advantage = ret-value
        a_hat = (advantage-advantage.mean())/advantage.std()

        # compute probability ratio
        mu_pi, logvar_pi = self.pi(state)
        dist_pi = Normal(mu_pi, logvar_pi.exp().sqrt())
        pi_logprob = dist_pi.log_prob(action)
        ratio = (pi_logprob-beta_logprob).sum(dim=1, keepdim=True).exp()
        optim.zero_grad()
        actor_loss = -torch.min(ratio*a_hat, torch.clamp(ratio, 1-self.eps, 1+self.eps)*a_hat).sum()
        critic_loss = advantage.pow(2).sum()
        loss = actor_loss+critic_loss
        self.hard_update(self.beta, self.pi)
        loss.backward()
        optim.step()
        del self.state[:]
        del self.action[:]
        del self.log_prob[:]
        del self.next_state[:]
        del self.reward[:]
        return loss.item()

Transition = namedtuple('Transition', ['state', 'action', 'next_state'])
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1)%self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

        

        