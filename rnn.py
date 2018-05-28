"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
matplotlib
numpy
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 30      # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.02           # learning rate

# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)    # float32 for converting torch FloatTensor
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)
    
    def R1(self, zeta, v):
        phi = zeta[0,0]
        theta = zeta[0,1]
        psi = zeta[0,2]

        R_z = self.Tensor([[psi.cos(), -psi.sin(), 0],
                            [psi.sin(), psi.cos(), 0],
                            [0., 0., 1.]])
        R_y = self.Tensor([[theta.cos(), 0., theta.sin()],
                            [0., 1., 0.],
                            [-theta.sin(), 0, theta.cos()]])
        R_x =  self.Tensor([[1., 0., 0.],
                            [0., phi.cos(), -phi.sin()],
                            [0., phi.sin(), phi.cos()]])
        R = torch.matmul(R_z, torch.matmul(R_y, R_x))
        return torch.matmul(R, torch.t(v)).view(1,-1)

    def R2(self, zeta, w):
        theta = zeta[0,1]
        psi = zeta[0,2]

        x11 = psi.cos()/theta.cos()
        x12 = psi.sin()/theta.cos()
        x13 = 0
        x21 = -psi.sin()
        x22 = psi.cos()
        x23 = 0
        x31 = psi.cos()*theta.tan()
        x32 = psi.sin()*theta.tan()
        x33 = 1
        R = self.Tensor([[x11, x12, x13],
                        [x21, x22, x23],
                        [x31, x32, x33]])
        return torch.matmul(R, torch.t(w)).view(1,-1)

    def transition(self, x0, state_action, dt):
        # state_action is [sin(zeta), cos(zeta), v, w, a]
        xyz = x0.clone()
        uvw_pqr = state_action[:,6:12].clone()
        zeta = state_action[:,0:3].asin()
        uvw = uvw_pqr[:,0:3]
        pqr = uvw_pqr[:,3:]
        uvw_dot = self.lin_accel(state_action)*self.uvw_dot_norm
        pqr_dot = self.ang_accel(state_action)*self.pqr_dot_norm
        dv, dw = uvw_dot*dt, pqr_dot*dt
        uvw = uvw+dv
        pqr = pqr+dw
        xyz_dot = self.R1(zeta, uvw)
        zeta_dot = self.R2(zeta, pqr)
        dx, dzeta = xyz_dot*dt, zeta_dot*dt
        xyz = xyz+dx
        zeta = zeta+dzeta
        return xyz, zeta, uvw, pqr

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # return outs, h_state

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot

for step in range(60):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    prediction, h_state = rnn(x, h_state)   # rnn output
    # !! next step is important !!
    h_state = Variable(h_state.data)        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # cross entropy loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()