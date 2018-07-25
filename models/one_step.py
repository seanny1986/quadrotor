import torch
import torch.nn.functional as F

class Transition(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, GPU=True):
        super(Transition,self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.pos = MLP(state_dim+action_dim, hidden_dim, 3, GPU)
        self.att = ATT(state_dim+action_dim, hidden_dim, 6, GPU)
        self.vel = MLP(state_dim+action_dim, hidden_dim, 3, GPU)
        self.ang = MLP(state_dim+action_dim, hidden_dim, 3, GPU)
        self.rpm = RPM(state_dim+action_dim, hidden_dim, 4, GPU)

        print(self.pos)
        print(self.att)
        print(self.vel)
        print(self.ang)
    
        self.opt = torch.optim.Adam(self.parameters(),lr=1e-4)
        
        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.pos = self.pos.cuda()
            self.att = self.att.cuda()
            self.vel = self.vel.cuda()
            self.ang = self.ang.cuda()
            self.rpm = self.rpm.cuda()
        else:
            self.Tensor = torch.Tensor

    def batch_update(self, traj):
        state_actions = traj["state_actions"]
        next_states = traj["next_states"]
        xyz_pred = self.pos(state_actions)
        att_pred = self.att(state_actions)
        vel_pred = self.vel(state_actions)
        ang_pred = self.ang(state_actions)
        RPM = self.rpm(state_actions)
        pred_state = torch.cat([xyz_pred, att_pred, vel_pred, ang_pred, RPM], dim=1)
        loss = F.mse_loss(pred_state, next_states)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
    
    def update(self, state_action, next_state):
        xyz_pred = self.pos(state_action)
        att_pred = self.att(state_action)
        vel_pred = self.vel(state_action)
        ang_pred = self.ang(state_action)
        RPM = self.rpm(state_action)
        pred_state = torch.cat([xyz_pred, att_pred, vel_pred, ang_pred, RPM], dim=1)
        loss = F.mse_loss(pred_state, next_state)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, GPU):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.GPU = GPU
        
        self.affine1 = torch.nn.Linear(input_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.affine1.weight)

        self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.affine2.weight)

        self.output_head = torch.nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.output_head.weight)

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))
        x = self.output_head(x)
        return x

class ATT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, GPU):
        super(ATT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.GPU = GPU
        
        self.affine1 = torch.nn.Linear(input_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.affine1.weight)

        self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.affine2.weight)

        self.output_head = torch.nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.output_head.weight)

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))
        x = self.output_head(x)
        return F.tanh(x)

class RPM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, GPU):
        super(RPM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.GPU = GPU
        
        self.affine1 = torch.nn.Linear(input_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.affine1.weight)

        self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.affine2.weight)

        self.output_head = torch.nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.output_head.weight)

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))
        x = self.output_head(x)
        return F.sigmoid(x)