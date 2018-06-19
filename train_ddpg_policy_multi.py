import environments.envs as envs 
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

env = envs.make("local_flight")

epochs = 250000
state_dim = 27
action_dim = 4
hidden_dim = 32

actor = ddpg.Actor(state_dim, action_dim)
target_actor = ddpg.Actor(state_dim, action_dim)
critic = ddpg.Critic(state_dim, action_dim)
target_critic = ddpg.Critic(state_dim, action_dim)
agent = ddpg.DDPG(actor,target_actor,critic,target_critic)

if args.cuda:
    agent = agent.cuda()

noise = OUNoise(action_dim)
noise.set_seed(args.seed)
memory = ddpg.ReplayMemory(1000000)

<<<<<<< HEAD:train_ddpg_policy_multiskill2.py
=======
vis = ani.Visualization(iris, 10)

def terminate(xyz, zeta, uvw, uvw_dot, rel_dist):
    mask1 = zeta[:2] > pi/2
    mask2 = zeta[:2] < -pi/2
    mask3 = np.abs(xyz) > 6
    if np.sum(mask1) > 0 or np.sum(mask2) > 0 or np.sum(mask3) > 0:
        return True
    if (rel_dist**2).sum() < dist_thresh:
        return True

last_xyz_dist = None
last_zeta_dist = None
last_uvw_dist = None
last_pqr_dist = None
def reward(rel_xyz, rel_zeta, rel_uvw, rel_pqr, action):
    
    global last_xyz_dist
    global last_zeta_dist
    global last_uvw_dist
    global last_pqr_dist
    
    err_xyz = (rel_xyz**2).mean()
    err_zeta = (rel_zeta**2).mean()
    err_uvw = (rel_uvw**2).mean()
    err_pqr = (rel_pqr**2).mean()
    
    xyz_loss = last_xyz_dist-err_xyz
    zeta_loss = last_zeta_dist-err_zeta
    uvw_loss = last_uvw_dist-err_uvw
    pqr_loss = last_pqr_dist-err_pqr
    
    last_xyz_dist = err_xyz
    last_zeta_dist = err_zeta
    last_uvw_dist = err_uvw
    last_pqr_dist = err_pqr
    
    action_rew = -(action**2).sum()/400000.
    
    bonus = 0.
    if err_xyz < dist_thresh:
        bonus = 1000.
    
    dist_rew = (-err_xyz).exp()+(-err_zeta).exp()+(-err_uvw).exp()+(-err_pqr).exp()
    ctrl_rew = xyz_loss+zeta_loss+uvw_loss+pqr_loss 
    return dist_rew+ctrl_rew+action_rew+bonus

def numpy_to_pytorch(xyz, zeta, uvw, pqr):
    xyz = torch.from_numpy(xyz.T).float()
    zeta = torch.from_numpy(zeta.T).float()
    uvw = torch.from_numpy(uvw.T).float()
    pqr = torch.from_numpy(pqr.T).float()
    if args.cuda:
        xyz = xyz.cuda()
        zeta = zeta.cuda()
        uvw = uvw.cuda()
        pqr = pqr.cuda()
    return xyz, zeta, uvw, pqr

def generate_goal(r):
    sph = Tensor(2).uniform_(-2*pi, 2*pi)
    x = r*sph[1].sin()*sph[0].cos()
    y = r*sph[1].sin()*sph[0].sin()
    z = r*sph[1].cos()
    xyz = Tensor([[x, y, z]])
    zeta = Tensor([[0.,0.,0.]])
    uvw = Tensor([[0.,0.,0.]])
    pqr = Tensor([[0.,0.,0.]])
    return xyz, zeta, uvw, pqr

def render(axis3d, goal, t):
    pl.figure(0)
    axis3d.cla()
    vis.draw3d(axis3d)
    vis.draw_goal(axis3d, goal)
    axis3d.set_xlim(-3, 3)
    axis3d.set_ylim(-3, 3)
    axis3d.set_zlim(0, 6)
    axis3d.set_xlabel('West/East [m]')
    axis3d.set_ylabel('South/North [m]')
    axis3d.set_zlabel('Down/Up [m]')
    axis3d.set_title("Time %.3f s" %(t*dt))
    pl.pause(0.001)
    pl.draw()

>>>>>>> 6a6fbf7af54d7a4b8571a3cca1c223da8d96442a:train_ddpg_policy_multi.py
def main():
    interval_avg = []
    avg = 0
    for ep in count(1):
<<<<<<< HEAD:train_ddpg_policy_multiskill2.py
        noise.reset()
        running_reward = 0
        state = env.reset()
        for t in range(env.T):            
=======

        # reset to [0,0,0], [0,0,0], [0,0,0], [0,0,0]
        xyz, zeta, uvw, pqr = iris.reset()
        xyz, zeta, uvw, pqr = numpy_to_pytorch(xyz, zeta, uvw, pqr)
        state = torch.cat([zeta.sin(), zeta.cos(), uvw, pqr],dim=1)

        # generate random goal state
        radius = Tensor(1).uniform_(0,max_rad)
        xyz_g, zeta_g, uvw_g, pqr_g = generate_goal(radius) 
        rel_dist = xyz_g-xyz
        rel_ang = zeta_g-zeta
        rel_uvw = uvw_g-uvw
        rel_pqr = pqr_g-pqr
        goal_init = torch.cat([rel_dist, rel_ang.sin(), rel_ang.cos(), rel_uvw, rel_pqr], dim=1)
        
        # reset noise, last distance to None, running reward to zero
        last_xyz_dist = (rel_dist**2).mean()
        last_zeta_dist = (rel_ang**2).mean()
        last_uvw_dist = (rel_uvw**2).mean()
        last_pqr_dist = (rel_pqr**2).mean()
        noise.reset()
        running_reward = 0
        for t in range(steps):
            
            # initialize goal to relative distance
            if t == 0:
                goal = goal_init
>>>>>>> 6a6fbf7af54d7a4b8571a3cca1c223da8d96442a:train_ddpg_policy_multi.py
            
            # render the episode
            if ep % args.log_interval == 0:
                env.render()
            
            # select an action using either random policy or trained policy
            if ep < args.warmup:
                action = agent.random_action(noise).data
            else:
                action = agent.select_action(state, noise=noise).item()
            
            # step simulation forward
            next_state, reward, done, _ = env.step(action)

            # push to replay memory
            memory.push(state, action, next_state, reward)
            
            # online training if out of warmup phase
            if ep >= args.warmup:
                for i in range(5):
                    transitions = memory.sample(args.batch_size)
                    batch = ddpg.Transition(*zip(*transitions))
                    agent.update(batch)
            
            # check if terminate
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