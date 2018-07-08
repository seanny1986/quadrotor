import policies.fmis as fmis
import argparse
import torch
import torch.nn.functional as F
from itertools import count
import environments.envs as envs
import os, gc

parser = argparse.ArgumentParser(description='PyTorch MBPS Node')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--batch-size', type=int, default=64, metavar='bs', help='training batch size')
parser.add_argument('--load', default=False, type=bool, metavar='l', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', type=bool, default=True, metavar='s', help='saves the model (default is True)')
parser.add_argument('--save-epochs', type=int, default=100, metavar='ep', help='save every n epochs (default 100)')
parser.add_argument('--load-path', type=str, default='', metavar='lp', help='load path string')
parser.add_argument('--cuda', type=bool, default=False, metavar='c', help='use CUDA for GPU acceleration (default True)')
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
action_bound = env.action_bound[1]

pi = fmis.Actor(state_dim, hidden_dim, action_dim, GPU=args.cuda)
beta = fmis.Actor(state_dim, hidden_dim, action_dim, GPU=args.cuda)
phi = fmis.Dynamics(state_dim+action_dim, hidden_dim, state_dim, GPU=args.cuda)
critic = fmis.Critic(state_dim, hidden_dim, 1, GPU=args.cuda)
agent = fmis.FMIS(pi, critic, beta, phi, env, GPU=args.cuda)

if args.cuda:
    agent = agent.cuda()

pi_optim = torch.optim.Adam(agent.parameters())
phi_optim = torch.optim.Adam(phi.parameters())

def main():
    interval_avg = []
    avg = 0
    #pid = os.getpid()
    #prev_mem=0
    for ep in count(1):
        running_reward = 0
        s_ = []
        a_ = []
        ns_ = []
        state = Tensor(env.reset())
        s0 = state.clone()
        #cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
        #add_mem = cur_mem - prev_mem
        #prev_mem = cur_mem
        #print("     train iterations: %s, added mem: %sM"%(ep, add_mem))
        for _ in range(env.H):
            if ep % args.log_interval == 0:
                env.render()          
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action[0].cpu().numpy())
            running_reward += reward
            next_state = Tensor(next_state)
            s_.append(state[0])
            a_.append(action[0])
            ns_.append(next_state[0])
            if done:
                break
            state = next_state
        trajectory = {"states": s_,
                    "actions": a_,
                    "next_states": ns_}
        for i in range(1):
            agent.model_update(pi_optim, trajectory)
        for i in range(1):
            policy_loss = agent.policy_update(phi_optim, s0, env.H)
            #print("Predicted Return: ", policy_loss)
        interval_avg.append(running_reward)
        avg = (avg*(ep-1)+running_reward)/ep   
        if ep % args.log_interval == 0:
            interval = float(sum(interval_avg))/float(len(interval_avg))
            print('Episode {}\t Interval average: {:.2f}\t Average reward: {:.2f}'.format(ep, interval, avg))
            interval_avg = []
            
if __name__ == '__main__':
    main()