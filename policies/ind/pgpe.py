import torch

class PGPE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.l1 = torch.nn.Linear(input_dim, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, output_dim)
        self.logvar = torch.nn.Linear(hidden_dim, output_dim)

    def set_weights(self, weights):
        
        # separate the weights for each layer
        l1_end = (self.input_dim*self.hidden_dim)+self.hidden_dim
        l1_W = torch.from_numpy(weights[:self.input_dim*self.hidden_dim].reshape(self.input_dim, self.hidden_dim))
        l1_b = torch.from_numpy(weights[self.input_dim*self.hidden_dim:l1_end])
        mu_W = torch.from_numpy(weights[l1_end:l1_end+(self.hidden_dim*self.output_dim)].reshape(self.hidden_dim, self.output_dim))
        mu_b = torch.from_numpy(weights[l1_end+(self.hidden_dim*self.output_dim):])
        logvar_W = torch.from_numpy(weights[l1_end:l1_end+(self.hidden_dim*self.output_dim)].reshape(self.hidden_dim, self.output_dim))
        logvar_b = torch.from_numpy(weights[l1_end+(self.hidden_dim*self.output_dim):])

        # set the weights for each layer
        self.fc1.weight.data.copy_(l1_W.view_as(self.l1.weight.data))
        self.fc1.bias.data.copy_(l1_b.view_as(self.l1.bias.data))
        self.mu.weight.data.copy_(mu_W.view_as(self.mu.weight.data))
        self.mu.bias.data.copy_(mu_b.view_as(self.mu.bias.data))
        self.logvar.weight.data.copy_(logvar_W.view_as(self.logvar.weight.data))
        self.logvar.bias.data.copy_(logvar_b.view_as(self.logvar.bias.data))
    
    def get_weights_dim(self):
        return (self.input_dim+1)*self.hidden_dim+(self.hidden_dim+1)*self.output_dim
    
    def forward(self, x):
        pass

    def select_action(self, state):
        pass

    def update(self):
        return