import torch
import random
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.distributions import Normal
from collections import namedtuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

"""
    Implements an architecture I'm calling Forward Model Importance Sampling. A statistical RNN 
    forward model is trained using the log-likelihood score function. The agent is trained under
    this model using PPO. 
"""

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
    def __init__(self, pi, beta, critic, env, network_settings, GPU=True):
        super(FMIS,self).__init__()
        self.pi = pi
        self.critic = critic
        self.beta = beta
        
        kern = C(1., (1e-3, 1e3))*RBF(1., (1e-2, 1e2))
        self.phi = GaussianProcessRegressor(kernel=kern, alpha=1e-3, n_restarts_optimizer=9)

        self.gamma = network_settings["gamma"]
        self.eps = network_settings["eps"]
        self.lmbd = network_settings["lambda"]
        self.GPU = GPU

        self.env = env
        self.observation_space = env.observation_space
        self.action_bound = env.action_bound[1]

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.pi = self.pi.cuda()
            self.critic = self.critic.cuda()
            self.beta = self.beta.cuda()
        else:
            self.Tensor = torch.Tensor

    def rollout(self, s0, H, samples=10):
        # rollout trajectory under model
        x = s0
        states = []
        actions = []
        next_states = []
        rewards = []
        log_probs = []
        for t in range(H):
            states.append(x)
            action, lp = self.select_action(x)
            state_action = torch.cat([x, action],dim=1).cpu().numpy()
            next_state = self.phi.predict(state_action)
            next_state = torch.from_numpy(next_state)
            if self.GPU:
                next_state = next_state.cuda().float()
            xyz = next_state[:,0:3].cpu().numpy().T
            zeta = (torch.atan2(next_state[:,3:6], next_state[:,6:9])).cpu().numpy().T
            uvw = next_state[:,9:12].cpu().numpy().T
            pqr = next_state[:,12:15].cpu().numpy().T
            a = action.cpu().numpy()[0]
            r = sum(self.env.reward((xyz,zeta,uvw,pqr), a))
            next_states.append(next_state)
            actions.append(action)
            log_probs.append(lp)
            rewards.append(self.Tensor([r]))
            x = next_state
        observations = {"states": states,
                        "actions": actions,
                        "log_probs": log_probs,
                        "next_states": next_states,
                        "rewards": rewards}
        return observations

    def select_action(self, x):
        mu, logvar = self.beta(x)
        min_sigma = torch.ones(x.size()[0], self.beta.output_dim)*1e-3
        if self.GPU:
            min_sigma = min_sigma.cuda()
        sigma = logvar.exp().sqrt()+min_sigma
        a = Normal(mu, sigma)
        action = a.sample()
        log_prob = a.log_prob(action)
        return F.sigmoid(action).pow(0.5), log_prob

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def model_update(self, memory):
        data = Transition(*zip(*memory.pull()))
        states = torch.stack(data.state)
        actions = torch.stack(data.action)
        next_states = torch.stack(data.next_state)
        X = torch.cat([states, actions],dim=1).cpu().numpy()
        y = next_states.cpu().numpy()
        print("--- Model fit ---")
        self.phi.fit(X, y)

    def policy_update(self, optim, s0, H):

        # rollout trajectory under statistical model
        trajectory = self.rollout(s0.detach(), H)
        state = torch.stack(trajectory["states"]).squeeze(1).detach()
        action = torch.stack(trajectory["actions"]).squeeze(1).detach()
        next_state = torch.stack(trajectory["next_states"]).squeeze(1).detach()
        reward = torch.stack(trajectory["rewards"]).detach()
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

        # PPO update
        actor_loss = -torch.min(ratio*a_hat, torch.clamp(ratio, 1-self.eps, 1+self.eps)*a_hat).sum()
        critic_loss = advantage.pow(2).sum()
        loss = actor_loss+critic_loss
        self.hard_update(self.beta, self.pi)
        loss.backward()
        optim.step()

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
        if len(self.memory)<batch_size:
            return self.memory
        else:
            lst = random.sample(self.memory, batch_size)
            return lst
    
    def pull(self):
        return self.memory

    def __len__(self):
        return len(self.memory)

        

        