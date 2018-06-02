import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from math import sin, cos, tan, atan2, asin

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

        self.zero = self.Tensor([[0.]])
        self.conj = self.Tensor([[1.],
                                [-1.],
                                [-1.],
                                [-1.]])

    def transition(self, x0, q, state_action, dt):
        # state_action is [sin(zeta), cos(zeta), v, w, a]
        xyz = x0.view(-1,1)
        q = q.view(-1,1)
        zeta = state_action[:,0:3].asin().view(-1,1)
        uvw_next = self.lin_vel(state_action).view(-1,1)
        pqr_next = self.ang_vel(state_action).view(-1,1)
        uvw_q = torch.cat([self.zero, uvw_next],dim=0)
        q_inv = self.q_conj(q)
        xyz_dot = torch.mm(self.q_mult(q_inv), torch.mm(self.q_mult(uvw_q), q))
        xyz = xyz+xyz_dot[1:]*dt
        zeta = self.q_to_euler(q)
        return xyz.view(1,-1), zeta.view(1,-1), uvw_next.view(1,-1), pqr_next.view(1,-1)
    
    def q_mult(self, p):
        """
            One way to compute the Hamilton product is usin Q(p)q, where Q(p) is
            the below 4x4 matrix, and q is a 4x1 quaternion. I decided not to do
            the full multiplication here, and instead return Q(p).  
        """

        p0, p1, p2, p3 = p[0,0], p[1,0], p[2,0], p[3,0]
        return self.Tensor([[p0, -p1, -p2, -p3],
                            [p1, p0, -p3, p2],
                            [p2, p3, p0, -p1],
                            [p3, -p2, p1, p0]])

    def q_conj(self, q):
        """
            Returns the conjugate of quaternion q. q* = q'/|q|, where q is the
            magnitude, and q' is the inverse [p0, -p1, -p2, -p3]^T. Since we
            always normalize after updating q, we should always have a unit
            quaternion. This means we don't have to normalize in this routine.
        """

        q[1:,:] = -q[1:,:]
        return q
    
    def q_to_euler(self, q):
        """
            Convert quaternion q to a set of angles zeta. We do all of the heavy
            lifting with quaternions, and then return the Euler angles since they
            are more intuitive.
        """

        q0, q1, q2, q3 = q[0,0], q[1,0], q[2,0], q[3,0]
        phi = atan2(2.*(q0*q1+q2*q3),q0**2-q1**2-q2**2+q3**2)
        theta = asin(2.*q0*q2-q3*q1)
        psi = atan2(2.*(q0*q3+q1*q2),q0**2+q1**2-q2**2-q3**2)
        return self.Tensor([[phi],
                            [theta],
                            [psi]])

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