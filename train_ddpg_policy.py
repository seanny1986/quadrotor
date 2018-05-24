import simulation.quadrotor as quad
import simulation.animation as ani
import simulation.config as cfg
import math
import numpy as np
import matplotlib.pyplot as pl
import policies.ddpg as ddpg
import argparse
from ounoise import OUNoise

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

noise = OUNoise(env.action_space.shape[0])
noise.set_seed(args.seed)

epochs = 250000
state_dim = 12
action_dim = 4
hidden_dim = 256

params = cfg.params

hover_thrust = (params["mass"]*9.81)/4.0
hover_rpm = math.sqrt(hover_thrust/params["kt"])
trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
iris = quad.Quadrotor(params)

actor = ddpg.Actor()
target_actor = ddpg.Actor()
critic = ddpg.Critic()
target_critic = ddpg.Critic()
agent = ddpg.DDPG(actor,target_actor,critic,target_critic)
pol_opt = optim.Adam(actor.parameters(), lr=1e-4)
crit_opt = optim.Adam(critic.parameters(), lr=1e-4)
memory = ddpg.ReplayMemory(1000000)

def main():
    interval_avg = []
    avg = 0
    for ep in count(1):
        state = env.reset()
        noise.reset()
        running_reward = 0
        for t in range(2500):
            if ep < args.warmup:
                action = agent.random_action(noise)
                action = action.data[0]
            else:    
                action = agent.select_action(torch.Tensor(state),noise=noise)
                action = action.data
            next_state, reward, done, _ = env.step(action.numpy())
            running_reward += reward
            memory.push(state.tolist(), action, next_state.tolist(), reward)
            if ep >= args.warmup:
                for i in range(5):               
                    transitions = memory.sample(args.batch_size)
                    batch = ddpg.Transition(*zip(*transitions))
                    agent.update(batch)
            if done:
                break
            state = next_state
        interval_avg.append(running_reward)
        avg = (avg*(ep-1)+running_reward)/ep   
        if ep % args.log_interval == 0:
            interval = float(sum(interval_avg))/float(len(interval_avg))
            print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
            interval_avg = []
            
if __name__ == '__main__':
    main()