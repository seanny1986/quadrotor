import policies.ppo as ppo
import environments.envs as envs
import argparse
import torch
import torch.nn.functional as F
from itertools import count

parser = argparse.ArgumentParser(description='PyTorch PPO hover experiment')
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

env = envs.make("hover")

epochs = 250000
state_dim = env.observation_space
action_dim = env.action_space
hidden_dim = 32

actor = ppo.Actor(state_dim, action_dim)
target_actor = ppo.Actor(state_dim, action_dim)
critic = ppo.Critic(state_dim, action_dim)
agent = ppo.PPO(actor,target_actor,critic)

if args.cuda:
    agent = agent.cuda()

def main():
    interval_avg = []
    avg = 0
    for ep in count(1):
        running_reward = 0
        state = env.reset()
        for t in range(env.T):
            if ep % args.log_interval == 0:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            running_reward += reward
            if done:
                break
            state = next_state
        agent.update()
        interval_avg.append(running_reward)
        avg = (avg*(ep-1)+running_reward)/ep   
        if ep % args.log_interval == 0:
            interval = float(sum(interval_avg))/float(len(interval_avg))
            print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
            interval_avg = []
            
if __name__ == '__main__':
    main()