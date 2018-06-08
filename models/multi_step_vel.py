from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import utils
from math import atan2, asin
import torch.nn.utils as ut

class Transition(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, GPU=True):
        super(Transition, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.uvw_out = nn.Linear(hidden_dim, output_dim)
        self.pqr_out = nn.Linear(hidden_dim, output_dim)

        self.GPU = GPU

        if GPU:
            self.Tensor = torch.cuda.FloatTensor

    def step(self, x0, zeta, uvw, pqr, dt):
        # state_action is [sin(zeta), cos(zeta), v, w, a]
        xyz = x0.view(-1,1)
        q = self.euler_to_q(zeta)
        q = q.view(-1,1)
        xyz_dot, q_dot = self.rotate(uvw, pqr, q)
        q = (q+q_dot*dt).norm(dim=0)
        xyz = xyz+xyz_dot[1:]*dt
        zeta = self.q_to_euler(q)
        return xyz.view(1,-1), zeta.view(1,-1)
        
    def rotate(self, uvw, pqr, q):
        uvw_q = torch.cat([self.zero, uvw],dim=0)
        pqr_q = torch.cat([self.zero, pqr],dim=0)
        q_inv = self.q_conj(q)
        xyz_dot = torch.mm(self.q_mult(q_inv), torch.mm(self.q_mult(uvw_q), q))
        q_dot = -0.5*self.q_mult(q).dot(pqr_q)
        return xyz_dot, q_dot

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
    
    def euler_to_q(self, zeta):
        return None

    def forward(self, state_action, H):
        outputs = []
        h_t = torch.zeros(1, self.hidden_dim, dtype=torch.float).cuda()
        c_t = torch.zeros(1, self.hidden_dim, dtype=torch.float).cuda()
        for i in range(H-1):
            h_t, c_t = self.lstm(state_action[i], (h_t, c_t))
            uvw = self.uvw_out(h_t)
            pqr = self.pqr_out(h_t)
            outputs.append(torch.cat([uvw, pqr],dim=1))
        return outputs

    def update(self, optimizer, criterion, xyzs, zetas, uvws, pqrs, actions):

        xs, ys = [], []
        H = len(xyzs)
        i = 0
        print("Update uvw: ", uvws)
        input("Paused")
        # process data
        for xyz, zeta, uvw, pqr in zip(xyzs, zetas, uvws, pqrs):   
            _, zeta_nn, uvw_nn, pqr_nn = utils.numpy_to_pytorch(xyz, zeta, uvw, pqr)
            if i < H-1:
                action = actions[i].reshape((1,-1))
                action = torch.from_numpy(action).float()
                if self.GPU:
                    action = action.cuda()
                state = torch.cat([zeta_nn.sin(), zeta_nn.cos(), uvw_nn, pqr_nn],dim=1)
                state_action = torch.cat([state, action],dim=1)
                xs.append(state_action)
            if i > 0:
                velocities = torch.cat([uvw_nn, pqr_nn],dim=1)
                ys.append(velocities)
            i += 1

        # update
        optimizer.zero_grad()
        ys_pred = self.forward(xs, H)
        ys_pred = torch.stack(ys_pred)
        ys = torch.stack(ys)
        loss = criterion(ys_pred, ys)
        print("PREDICTED: ", ys_pred)
        print("ACTUAL: ",ys)
        print("LOSS: ", loss.item())
        input("Paused")
        loss.backward()
        ut.clip_grad_norm_(self.parameters(),0.1)
        optimizer.step()
        return loss.item()