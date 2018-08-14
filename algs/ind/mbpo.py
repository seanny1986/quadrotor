import torch
import torch.nn as nn
import torch.nn.functional as F

class Dynamics(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Dynamics,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return self.fc2(x)

class DeterministicPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeterministicPolicy,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return self.fc2(x)

class PolicySearch(nn.Module):
    def __init__(self, policy, dynamics):
        super(PolicySearch,self).__init__()
        self.policy = policy
        self.dynamics = dynamics
        self.dyn_opt = torch.optim.Adam(self.dynamics.Parameters())
    
    def select_action(self, state):
        return self.policy(state)

    def update_model(self, batch):
        states = torch.stack(batch["states"])
        actions = torch.stack(batch["actions"])
        next_states = torch.stack(batch["next_states"])
        state_actions = torch.cat([states, actions], dim=1)
        deltas = next_states-states
        pred_deltas = self.dynamics(state_actions)
        loss = F.mse_loss(pred_deltas, deltas)
        self.dyn_opt.zero_grad()
        loss.backward()
        self.dyn_opt.step()

    def update_policy(self):
        pass

