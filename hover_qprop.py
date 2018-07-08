import policies.qprop as qprop
import argparse
import torch
import torch.nn.functional as F
from itertools import count
import environments.envs as envs
import utils

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

env = envs.make("hover")

state_dim = env.observation_space
action_dim = env.action_space
hidden_dim = 32

actor = qprop.Actor(state_dim, hidden_dim, action_dim)
target_actor = qprop.Actor(state_dim, hidden_dim, action_dim)
critic = qprop.Critic(state_dim+action_dim, hidden_dim, action_dim)
target_critic = qprop.Critic(state_dim+action_dim, hidden_dim, action_dim)
memory = qprop.Memory(1000000)
agent = qprop.QPROP(actor, critic, memory, target_actor, target_critic, env)

if args.cuda:
    agent = agent.cuda()

action_bound = env.action_bound[1]

def main():
    interval_avg = []
    avg = 0
    for ep in count(1):
        running_reward = 0
        state = Tensor(env.reset())
        states = []
        actions = []
        rewards = []
        log_probs = []
        for t in range(env.H):
            if ep % args.log_interval == 0:
                env.render()          
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.data[0].cpu().numpy())
            running_reward += reward
            next_state = Tensor(next_state)
            reward = Tensor([reward])
            memory.push(state[0], action[0], next_state[0], reward)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            if ep >= args.warmup:
                for i in range(3):               
                    transitions = memory.sample(args.batch_size)
                    batch = qprop.Transition(*zip(*transitions))
                    agent.online_update(batch)
            if done:
                break
            state = next_state
        trajectory = {"states": states,
                        "actions": actions,
                        "rewards": rewards,
                        "log_probs": log_probs}
        agent.offline_update(trajectory)
        interval_avg.append(running_reward)
        avg = (avg*(ep-1)+running_reward)/ep   
        if ep % args.log_interval == 0:
            interval = float(sum(interval_avg))/float(len(interval_avg))
            print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
            interval_avg = []
            
if __name__ == '__main__':
    main()