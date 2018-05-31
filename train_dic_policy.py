import random
import argparse
import torch
import policies.dic
import mbps_utils
import csv
from torch.autograd import Variable
from mpl_toolkits.mplot3d import axes3d 
import matplotlib.pyplot as plt
import matplotlib.style as style
import torch.nn.functional as F
import simulation.quadrotor as quad
import simulation.config as cfg
import simulation.animation as ani
import models.one_step_velocity as model
import policies.dic as dic

parser = argparse.ArgumentParser(description='PyTorch MBPS Node')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--warmup', type=int, default=100, metavar='w', help='number of warmup episodes')
parser.add_argument('--batch-size', type=int, default=64, metavar='bs', help='training batch size')
parser.add_argument('--load', default=False, type=bool, metavar='l', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', type=bool, default=True, metavar='s', help='saves the model (default is True)')
parser.add_argument('--save-epochs', type=int, default=10, metavar='ep', help='save every n epochs (default 100)')
parser.add_argument('--load-path', type=str, default='', metavar='lp', help='load path string')
parser.add_argument('--cuda', type=bool, default=True, metavar='c', help='use CUDA for GPU acceleration (default True)')
parser.add_argument('--plot-interval', type=int, default=100, metavar='pi', help='interval between plot updates')
args = parser.parse_args()

if args.cuda:
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.Tensor

state_dim = 12
action_dim = 4
hidden_dim = 32
goal_dim = 12
epochs = 1500

dyn = model.Transition(state_dim, action_dim, hidden_dim)
pol = dic.FeedForwardPolicy(state_dim+goal_dim, hidden_dim, action_dim, dyn, args.cuda)
pol_opt = torch.optim.Adam(pol.parameters(), lr=1e-4)

memory = dic.ReplayMemory(1000000)

with open('maneuvers.csv', 'r') as f:
  reader = csv.reader(f)
  maneuvers = list(reader)

maneuver_list = []
for i, m in enumerate(maneuvers):
    if i > 0:
        x0 = [float(x) for x in m[0:16]]
        g0 = [float(g) for g in m[16:]]
        maneuver_list.append([x0, g0])
maneuvers = maneuver_list

plt.close("all")
plt.ion()
fig = plt.figure(0)
axis3d = fig.add_subplot(111, projection='3d')
params = cfg.params
iris = quad.Quadrotor(params)
dt = iris.dt
vis = ani.Visualization(iris,10)

def main():                                                                        
    
    for i in range(epochs):
        # run policy on the aircraft
        run_policy(maneuvers, True)

        # optimization of policy under the model
        train_policy(maneuvers)
        print()
        input("Press any key to continue")

        # reset environment
        iris.reset()

    # save policy
    mbps_utils.save(pol, filename="dic_policy.pth.tar")
    mbps_utils.save(dyn, filename="one_step_vel_model.pth.tar")

def train_policy(maneuvers):
    T = int(g[-1])
    for i in range(T):
        pass
    

def run_policy(maneuvers, set_state=True):
    for m in maneuvers:
        xyz_init = m[0][0:3]
        zeta_init = m[0][3:6]
        uvw_init = m[0][6:9]
        pqr_init = [0][9:12]
        
        xyz_g = m[1][0:3]
        zeta_g = m[1][3:6]
        uvw_g = m[1][6:9]
        pqr_g = m[1][9:12]

        if set_state:
            iris.set_state(xyz_init, zeta_init, uvw_init, pqr_init)
        xyz, zeta, uvw, pqr = iris.get_state()
        state = torch.cat([zeta.sin(), zeta.cos(), uvw, pqr], dim=1)
        T = int(g[-1])
        for t in range(T):
            dist_xyz = xyz_g-xyz
            dist_zeta = zeta_g-zeta
            dist_uvw = uvw_g-uvw
            dist_pqr = pqr_g-pqr
            goal = torch.cat([dist_xyz, dist_zeta, dist_uvw, dist_pqr],dim=1)
            action = pol.select_action(state, goal)
            xyz_next, zeta_next, uvw_next, pqr_next = iris.step(action)
            next_state = torch.cat([zeta_next.sin(), zeta_next.cos(), uvw_next, pqr_next],dim=1)
            memory.push(state.squeeze(0), action.squeeze(0), next_state.squeeze(0))
            for i in range(5):
                transitions = memory.sample(args.batch_size)
                batch = dic.Transition(*zip(*transitions))
                dyn.batch_update(state, action, next_state)
            state = next_state
            xyz = xyz_next
            zeta = zeta_next
            uvw = uvw_next
            pqr = pqr_next

        print("Maneuver Loss: {}".format(((state-Tensor(g[:-1])).pow(2)).mean()))

if __name__ == "__main__":
    main()
