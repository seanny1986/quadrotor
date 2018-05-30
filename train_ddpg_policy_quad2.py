import simulation.quadrotor2 as quad
import simulation.animation as ani
import simulation.config as cfg
from math import sqrt, pi
import numpy as np
import matplotlib.pyplot as pl
import policies.ddpg as ddpg
import argparse
from ounoise import OUNoise
import torch
import torch.nn.functional as F
from itertools import count

parser = argparse.ArgumentParser(description='PyTorch MBPS Node')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--warmup', type=int, default=100, metavar='w', help='number of warmup episodes')
parser.add_argument('--batch-size', type=int, default=64, metavar='bs', help='training batch size')
parser.add_argument('--load', default=False, type=bool, metavar='l', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', type=bool, default=True, metavar='s', help='saves the model (default is True)')
parser.add_argument('--save-epochs', type=int, default=100, metavar='ep', help='save every n epochs (default 100)')
parser.add_argument('--load-path', type=str, default='', metavar='lp', help='load path string')
parser.add_argument('--cuda', type=bool, default=True, metavar='c', help='use CUDA for GPU acceleration (default True)')
parser.add_argument('--plot-interval', type=int, default=100, metavar='pi', help='interval between plot updates')
args = parser.parse_args()

if args.cuda:
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.Tensor

epochs = 250000
state_dim = 12
action_dim = 4
hidden_dim = 32

params = cfg.params
iris = quad.Quadrotor(params)
hover_rpm = iris.hov_rpm
trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
action_bound = iris.max_rpm
dt = iris.dt
T = 15
steps = int(T/dt)

actor = ddpg.Actor(state_dim, action_dim)
target_actor = ddpg.Actor(state_dim, action_dim)
critic = ddpg.Critic(state_dim, action_dim)
target_critic = ddpg.Critic(state_dim, action_dim)
agent = ddpg.DDPG(action_bound, actor,target_actor,critic,target_critic)

if args.cuda:
    agent = agent.cuda()

noise = OUNoise(action_dim)
noise.set_seed(args.seed)
memory = ddpg.ReplayMemory(1000000)

goal = Tensor([[0., 0., 3.], 
                [0., 0., 0.]])

vis = ani.Visualization(iris, 10, quaternion=True)

def reward(xyz, zeta):
    return -10.*F.mse_loss(xyz[0], goal[0,:])-30.*F.mse_loss(zeta[0], goal[1,:])

def main():
    pl.close("all")
    pl.ion()
    fig = pl.figure(0)
    axis3d = fig.add_subplot(111, projection='3d')

    interval_avg = []
    avg = 0
    for ep in count(1):
        noise.reset()
        running_reward = 0
        xyz, zeta, q, uvw, pqr = iris.reset()
        xyz = torch.from_numpy(xyz.T).float()
        zeta = torch.from_numpy(zeta.T).float()
        uvw = torch.from_numpy(uvw.T).float()
        pqr = torch.from_numpy(pqr.T).float()
        if args.cuda:
            zeta = zeta.cuda()
            uvw = uvw.cuda()
            pqr = pqr.cuda()
        state = torch.cat([zeta.sin(), zeta.cos(), uvw, pqr],dim=1)
        for t in range(steps):
            if ep % args.log_interval == 0:
                pl.figure(0)
                axis3d.cla()
                vis.draw3d_quat(axis3d)
                axis3d.set_xlim(-3, 3)
                axis3d.set_ylim(-3, 3)
                axis3d.set_zlim(0, 6)
                axis3d.set_xlabel('West/East [m]')
                axis3d.set_ylabel('South/North [m]')
                axis3d.set_zlabel('Down/Up [m]')
                axis3d.set_title("Time %.3f s" %(t*dt))
                pl.pause(0.001)
                pl.draw()
            if ep < args.warmup:
                action = agent.random_action(noise)
                action = action.data
            else:    
                action = agent.select_action(state,noise=noise)
                action = action.data
            xyz, zeta, q, uvw, pqr = iris.step(action.cpu().numpy()[0])
            mask1 = zeta > pi/2
            mask2 = zeta < -pi/2
            mask3 = np.abs(xyz) > 6
            if np.sum(mask1) > 0 or np.sum(mask2) > 0 or np.sum(mask3) > 0:
                break
            xyz = torch.from_numpy(xyz.T).float()
            zeta = torch.from_numpy(zeta.T).float()
            uvw = torch.from_numpy(uvw.T).float()
            pqr = torch.from_numpy(pqr.T).float()
            if args.cuda:
                xyz = xyz.cuda()
                zeta = zeta.cuda()
                uvw = uvw.cuda()
                pqr = pqr.cuda()
            next_state = torch.cat([zeta.sin(), zeta.cos(), uvw, pqr],dim=1)
            r = reward(xyz, zeta)+100.-action.pow(2).sum()/1e6
            running_reward += r
            memory.push(state.squeeze(0), action.squeeze(0), next_state.squeeze(0), r.unsqueeze(0))
            if ep >= args.warmup:
                for i in range(5):               
                    transitions = memory.sample(args.batch_size)
                    batch = ddpg.Transition(*zip(*transitions))
                    agent.update(batch)
            state = next_state
        interval_avg.append(running_reward)
        avg = (avg*(ep-1)+running_reward)/ep   
        if ep % args.log_interval == 0:
            interval = float(sum(interval_avg))/float(len(interval_avg))
            print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
            interval_avg = []
            
if __name__ == '__main__':
    main()