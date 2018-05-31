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
    
        self.lin_vel_opt = torch.optim.Adam(self.lin_vel.parameters(),lr=1e-5)
        self.ang_vel_opt = torch.optim.Adam(self.ang_vel.parameters(),lr=1e-5)
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
        uvw_next = self.lin_vel(state_action)
        pqr_next = self.ang_vel(state_action)
        xyz_dot = torch.mm(self.R1(zeta), uvw_next.t()).t()
        zeta_dot = torch.mm(self.R2(zeta), pqr_next.t()).t()
        xyz = xyz+xyz_dot*dt
        zeta = zeta+zeta_dot*dt
        return xyz, zeta, uvw_next, pqr_next
    
    def batch_R1(self, zetas):
        phis = zetas[:,0]
        thetas = zetas[:,1]
        psis = zetas[:,2]
        one = torch.ones(phis.size())
        zero = torch.zeros(phis.size())
        R_z1 = torch.cat([psis.cos(),      -psis.sin(),          zero], dim=1)
        R_z2 = torch.cat([psis.sin(),      psis.cos(),           zero],dim=1)
        R_z3 = torch.cat([zero,                zero,             one],dim=1)
        R_z = torch.cat([R_z1, R_z2, R_z3], dim=2)
        R_y1 = torch.cat([thetas.cos(),        zero,     thetas.sin()],dim=1)
        R_y2 = torch.cat([zero,                one,             zero], dim=1)
        R_y3 = torch.cat([-thetas.sin(),       zero,     thetas.cos()], dim=1)
        R_y = torch.cat([R_y1, R_y2, R_y3],dim=2)
        R_x1 = torch.cat([one,               zero,             zero],dim=1)
        R_x2 = torch.cat([zero,            phis.cos(),       -phis.sin()],dim=1)
        R_x3 = torch.cat([zero,            phis.sin(),       phis.cos()],dim=1)
        R_x =  torch.cat([R_x1, R_x2, R_x3],dim=2)
        return torch.bmm(R_z, torch.bmm(R_y, R_x))
    
    def batch_R2(self, zetas):
        thetas = zetas[:,1]
        psis = zetas[:,2]
        one = torch.ones(phis.size())
        zero = torch.zeros(phis.size())
        R_1 = torch.cat([psis.cos()/thetas.cos(),      psis.cos()/thetas.cos(),             zero], dim=1)
        R_2 = torch.cat([-psis.sin(),                       psis.cos(),                     zero],dim=1)
        R_3 = torch.cat([psis.cos()*thetas.tan(),      psis.sin()*thetas.tan(),             one],dim=1)
        return torch.cat([R_1, R_2, R_3],dim=2)

    def batch_transition(self, x0s, state_actions, dt):
        xyzs = x0s
        zetas = state_actions[:,0:3].asin()
        uvw_next = self.lin_vel(state_actions).t()
        pqr_next = self.ang_vel(state_actions).t()
        xyz_dots = torch.bmm(self.batch_R1(zetas), uvw_next.unsqueeze(2)).t().squeeze(2)
        zeta_dots = torch.mm(self.batch_R2(zetas), pqr_next.unsqueeze(2)).t().squeeze(2)
        xyzs = xyzs+xyz_dots*dt
        zetas = zetas+zeta_dots*dt
        return xyzs, zetas, uvw_next, pqr_next

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
    
    def batch_update(self, batch):
        state = Variable(torch.stack(batch.state))
        action = Variable(torch.stack(batch.action))
        next_state = Variable(torch.stack(batch.next_state))
        reward = Variable(torch.cat(batch.reward))
        reward = torch.unsqueeze(reward, 1)
        state_action = torch.cat([state, action],dim=1)

        v_next = self.lin_vel(state_action)
        w_next = self.ang_vel(state_action)

        v_next_loss = F.mse_loss(v_next, next_state[6:9])
        w_next_loss = F.mse_loss(w_next, next_state[9:12])

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