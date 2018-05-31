import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from math import sin, cos, tan

class Transition(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, GPU=True):
        super(Transition, self).__init__()
        self.lin_vel = MLP(state_dim+action_dim, hidden_dim, 3, GPU)
        self.ang_vel = MLP(state_dim+action_dim, hidden_dim, 3, GPU)
    
        self.lin_vel_opt = torch.optim.Adam(self.lin_vel.parameters(),lr=1e-4)
        self.ang_vel_opt = torch.optim.Adam(self.ang_vel.parameters(),lr=1e-4)
        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
            self.lin_vel = self.lin_vel.cuda()
            self.ang_vel = self.ang_vel.cuda()
        else:
            self.Tensor = torch.Tensor

    def R1(self, zeta):
        """
            Rotation matrix converting body frame linear values to the inertial frame.
            This matrix is orthonormal, so to go from the inertial frame to the body
            frame, we can take the transpose of this matrix. That is, R1^-1 = R1^T.
            These rotations are for an East-North-Up axis system, since matplotlib 
            uses this for plotting. If you wanted to use N-E-D as is more typical in
            aerospace, you would need two additional rotation matrices for plotting -- 
            a pi/2 rotation about the inertial z-axis, and then another pi/2 rotation 
            about the inertial x-axis.
        """
        
        phi = zeta[0,0]
        theta = zeta[0,1]
        psi = zeta[0,2]
        
        R_z = self.Tensor([[cos(psi),      -sin(psi),          0.],
                            [sin(psi),      cos(psi),           0.],
                            [0.,                0.,             1.]])
        R_y = self.Tensor([[cos(theta),        0.,     sin(theta)],
                            [0.,                1.,             0.],
                            [-sin(theta),       0.,     cos(theta)]])
        R_x =  self.Tensor([[1.,               0.,             0.],
                            [0.,            cos(phi),       -sin(phi)],
                            [0.,            sin(phi),       cos(phi)]])
        return torch.mm(R_z,torch.mm(R_y, R_x))

    def R2(self, zeta):
        """
            Euler rates rotation matrix converting body frame angular velocities 
            to the inertial frame. This uses the East-North-Up axis convention, so 
            it looks a bit different to the rates matrix in most aircraft dynamics
            textbooks (which use an N-E-D system).
        """

        theta = zeta[0,1]
        psi = zeta[0,2]
        return self.Tensor([[cos(psi)/cos(theta), sin(psi)/cos(theta), 0.],
                            [-sin(psi),             cos(psi),           0.],
                            [cos(psi)*tan(theta), sin(psi)*tan(theta),  1.]])

    def transition(self, x0, state_action, dt):
        # state_action is [sin(zeta), cos(zeta), v, w, a]
        xyz = x0
        zeta = state_action[:,0:3].asin()
        uvw = state_action[:,6:9]
        pqr = state_action[:,9:]
        uvw_next = self.lin_vel(state_action)
        pqr_next = self.ang_vel(state_action)
        xyz_dot = torch.mm(self.R1(zeta), uvw_next.t()).t()
        zeta_dot = torch.mm(self.R2(zeta), pqr_next.t()).t()
        xyz = xyz+xyz_dot*dt
        zeta = zeta+zeta_dot*dt
        return xyz, zeta, uvw, pqr

    def update(self, zeta, uvw, pqr, action, uvw_next, pqr_next):
        zeta = zeta.reshape((1,-1))
        uvw = uvw.reshape((1,-1))
        pqr = pqr.reshape((1,-1))
        action = action.reshape((1,-1))
        uvw_next = uvw_next.reshape((1,-1))
        pqr_next = pqr_next.reshape((1,-1))
       
        zeta = torch.from_numpy(zeta).float()
        uvw = torch.from_numpy(uvw).float()
        pqr = torch.from_numpy(pqr).float()
        action = torch.from_numpy(action).float()
        uvw_next = torch.from_numpy(uvw_next).float()
        pqr_next = torch.from_numpy(pqr_next).float()

        if self.GPU:
            zeta = zeta.cuda()
            uvw = uvw.cuda()
            pqr = pqr.cuda()
            action = action.cuda()
            uvw_next = uvw_next.cuda()
            pqr_next = pqr_next.cuda()
        
        state = torch.cat([zeta.sin(), zeta.cos(), uvw, pqr],dim=1)
        state_action = torch.cat([state, action],dim=1)
        
        v_next = self.lin_vel(state_action)
        w_next = self.ang_vel(state_action)

        v_next_loss = F.mse_loss(v_next, uvw_next)
        w_next_loss = F.mse_loss(w_next, pqr_next)

        self.lin_vel_opt.zero_grad()
        self.ang_vel_opt.zero_grad()

        v_next_loss.backward()
        w_next_loss.backward()

        self.lin_vel_opt.step()
        self.ang_vel_opt.step()

        return v_next_loss.item(), w_next_loss.item()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, GPU):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.GPU = GPU
        
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.affine1.weight)

        self.output_head = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.output_head.weight)

        if GPU:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = self.output_head(x)
        return x