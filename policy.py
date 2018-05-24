import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal
import mbps_utils

"""
    Contains classes for the different types of policy we might want to run. At the moment, only a basic MLP
    policy has been implemented. Later improvements will include a recurrent policy, and possibly more advanced
    architectures depending on time/success.
"""

class FeedForwardPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, transition_model, GPU=True):
        super(FeedForwardPolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transition_model = transition_model
        self.GPU = GPU

        self.affine1 = nn.Linear(input_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.affine1.weight)

        self.mu = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.mu.weight)
        
        self.logvar = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.logvar.weight)

        self.alpha_1 = 1.0
        self.alpha_2 = 1e-6

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        mu = F.sigmoid(self.mu(x))
        logvar = self.logvar(x)
        return mu, logvar
    
    def select_action(self, state, goal, noise=True):
        x = torch.cat([state, goal],dim=1)
        mu, logvar = self.forward(x)
        if not noise:
            return mu
        else:
            eps = torch.randn(logvar.size())
            if self.GPU: 
                eps = eps.cuda()
            return mu+(logvar.exp()).sqrt()*Variable(eps)

    def random_action(self):
        eps = torch.rand(1, self.output_dim)
        if self.GPU:
            eps = eps.cuda()
        return eps

    def calc_dist_loss(self, state, goal):
        dist_loss = F.mse_loss(state, goal)
        return dist_loss

    def calc_reward(self, state, goal):
        # split into components for ease of sanity checking
        X, Xg = state[:,0:3], goal[:,0:3]
        PHI, PHIg = state[:,3:6], goal[:,3:6]
        V, Vg = state[:,6:9], goal[:,6:9]
        W, Wg = state[:,9:12], goal[:,9:12]

        # calculate loss from individual components so that we can weight these
        dist_loss = self.calc_dist_loss(X, Xg)
        att_loss = self.calc_dist_loss(PHI, PHIg)
        lin_vel_loss = self.calc_dist_loss(V, Vg)
        ang_vel_loss = self.calc_dist_loss(W, Wg)
        return dist_loss+att_loss+lin_vel_loss+ang_vel_loss

    def update(self, maneuvers, opt, dt):
        states = [m[0] for m in maneuvers]
        goals = [m[1] for m in maneuvers]
        states = Variable(self.Tensor(states))/self.transition_model.mu_state_max
        goals = Variable(self.Tensor(goals))
        steps = int(1.5/dt)
        dts = self.Tensor([dt]).expand(states.size()[0],-1)
        for _ in range(steps):
            states = torch.cat([states, dts],dim=1)
            actions = self.select_action(states, goals, False)
            state_actions = torch.cat([states, actions], dim=1)
            states = torch.cat(self.transition_model.transition(state_actions),dim=1)
        loss = self.calc_reward(states*self.transition_model.mu_state_max, goals[:,:-1])
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def set_transition_model(self, transition_model):
        self.transition_model = transition_model

class Recurrent(nn.Module):
    def __init__(self):
        pass