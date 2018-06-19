import torch
import torch.nn.functional as F

class Transition(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, GPU=True):
        super(Transition,self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.pos = MLP(state_dim+action_dim, hidden_dim, 3, GPU)
        self.att = MLP(state_dim+action_dim, hidden_dim, 3, GPU)

        print(self.pos)
        print(self.att)
    
        self.pos_opt = torch.optim.Adam(self.pos.parameters(),lr=1e-4)
        self.att_opt = torch.optim.Adam(self.att.parameters(),lr=1e-4)
        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.pos = self.pos.cuda()
            self.att = self.att.cuda()
        else:
            self.Tensor = torch.Tensor
    
    def transition(self, xyz, zeta, action):
        state = torch.cat([zeta.sin(), zeta.cos(), xyz],dim=1)
        state_action = torch.cat([state, action], dim=1)
        xyz = self.pos(state_action)
        zeta = self.att(state_action)
        return xyz, zeta

    def update(self, xyz, zeta, action, xyz_next, zeta_next):
        xyz = xyz.reshape((1,-1))
        zeta = zeta.reshape((1,-1))
        action = action.reshape((1,-1))
        xyz_next = xyz_next.reshape((1,-1))
        zeta_next = zeta_next.reshape((1,-1))
       
        xyz = torch.from_numpy(xyz).float()
        zeta = torch.from_numpy(zeta).float()     
        xyz_next = torch.from_numpy(xyz_next).float()
        zeta_next = torch.from_numpy(zeta_next).float()
        action = torch.from_numpy(action).float()
        
        if self.GPU:
            xyz = xyz.cuda()
            xyz_next = xyz_next.cuda()
            zeta = zeta.cuda()
            zeta_next = zeta_next.cuda()
            action = action.cuda()
        
        state = torch.cat([zeta.sin(), zeta.cos(), xyz],dim=1)
        state_action = torch.cat([state, action],dim=1)
        
        xyz_pred = self.pos(state_action)
        att_pred = self.att(state_action)

        xyz_loss = F.mse_loss(xyz_pred, xyz_next)
        zeta_loss = F.mse_loss(att_pred, zeta_next.cos())

        self.pos_opt.zero_grad()
        self.att_opt.zero_grad()

        xyz_loss.backward()
        zeta_loss.backward()

        self.pos_opt.step()
        self.att_opt.step()

        return xyz_loss.item(), zeta_loss.item()
            

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, GPU):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.GPU = GPU
        
        self.affine1 = torch.nn.Linear(input_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.affine1.weight)

        self.output_head = torch.nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.output_head.weight)

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = self.output_head(x)
        return x