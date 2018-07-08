import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def gen_observations(mean, std, samples=100):
    dist = Normal(mean, std)
    return dist.sample(samples)

mean = 10
std = 5

class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
    
    def sample_from_distribution(self, x, samples=1):
        mu, logvar = self.forward(x)
        dist = Normal(mu, logvar.exp().sqrt())
        pred = dist.sample(samples)
        logprob = dist.log_prob(pred)
        return pred, logprob
    
    def get_mean_and_variance(self, x):
        mu, logvar = self.forward(x)
        return mu, logvar.exp().sqrt()

    def update(self, batch, optimizer):
        xs = batch["xs"]
        ys = batch["ys"]
        pred_y, logprob = self.sample_from_distribution(xs)
        loss = logprob*(pred_y.detach()-ys)**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        