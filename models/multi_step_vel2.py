from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Transition(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(input_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

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

    def forward(self, x, future=0):
        outputs = []
        h_t = torch.zeros(x.size(0), self.hidden_dim, dtype=torch.float)
        c_t = torch.zeros(x.size(0), self.hidden_dim, dtype=torch.float)
        h_t2 = torch.zeros(x.size(0), 51, dtype=torch.float)
        c_t2 = torch.zeros(x.size(0), 51, dtype=torch.float)

        for i, input_t in enumerate(x.chunk(x.size(dim=1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, dim=1).squeeze(dim=2)
        return outputs

    def update(self, optimizer, criterion, xs, ys):
        def closure():
            optimizer.zero_grad()
            ys_pred = self.forward(xs)
            loss = criterion(ys_pred, ys)
            loss.backward()
            return loss
        optimizer.step(closure)


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])
    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
plt.close()