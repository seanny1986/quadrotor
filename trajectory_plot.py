import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
import gym_aero
import numpy as np
import utils
import argparse
import config as cfg
from math import sin, cos, tan, pi

"""
    Function to play back saved policies and save video. I hate matplotlib.

    -- Sean Morrison, 2018
"""
plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

# script arguments. E.g. python play_back.py --env="Hover" --pol="ppo"
parser = argparse.ArgumentParser(description="PyTorch actor-critic example")
parser.add_argument("--env", type=str, default="Hover", metavar="E", help="environment to run")
parser.add_argument("--pol", type=str, default="ppo", metavar="P", help="policy to run")
#parser.add_argument("--final", type=bool, default=False, metavar="F", help="load final policy? True/False")
parser.add_argument("--fname", type=str, default=None, metavar="F", help="load final policy? True/False")
parser.add_argument('--name', type=str, default='fig', metavar='N', help='name to save figure as')
args = parser.parse_args()

# animation callback function
def plot_traj(P, A, xyz, zeta, ax, g):
    
    # rotation matrix. Only used for plotting, has no effect on simulation calcs.
    def R1(zeta):
        phi = zeta[0,0]
        theta = zeta[1,0]
        psi = zeta[2,0]
        
        R_z = np.array([[cos(psi),      -sin(psi),          0.],
                        [sin(psi),      cos(psi),           0.],
                        [0.,                0.,             1.]])
        R_y = np.array([[cos(theta),        0.,     sin(theta)],
                        [0.,                1.,             0.],
                        [-sin(theta),       0.,     cos(theta)]])
        R_x =  np.array([[1.,               0.,             0.],
                        [0.,            cos(phi),       -sin(phi)],
                        [0.,            sin(phi),       cos(phi)]])
        return R_z.dot(R_y.dot(R_x))
    
    p1, p2, p3, p4 = P
    m1, m2, m3, m4 = A
    goal = g
    R = R1(zeta)
       
    # rotate to aircraft attitude
    q1 = np.einsum('ij,kj->ik', p1, R.T)
    q2 = np.einsum('ij,kj->ik', p2, R.T)
    q3 = np.einsum('ij,kj->ik', p3, R.T)
    q4 = np.einsum('ij,kj->ik', p4, R.T)
    
    n1 = np.einsum('ij,kj->ik', m1, R.T)
    n2 = np.einsum('ij,kj->ik', m2, R.T)
    n3 = np.einsum('ij,kj->ik', m3, R.T)
    n4 = np.einsum('ij,kj->ik', m4, R.T)

    # translate to aircraft position
    q1 = np.matlib.repmat(xyz.T,n+1,1)+q1
    q2 = np.matlib.repmat(xyz.T,n+1,1)+q2
    q3 = np.matlib.repmat(xyz.T,n+1,1)+q3
    q4 = np.matlib.repmat(xyz.T,n+1,1)+q4

    n1 = np.matlib.repmat(xyz.T,n+1,1)+n1
    n2 = np.matlib.repmat(xyz.T,n+1,1)+n2
    n3 = np.matlib.repmat(xyz.T,n+1,1)+n3
    n4 = np.matlib.repmat(xyz.T,n+1,1)+n4

    # plot rotated, translated points
    ax.plot(q1[:,0], q1[:,1], q1[:,2],'k')
    ax.plot(q2[:,0], q2[:,1], q2[:,2],'k')
    ax.plot(q3[:,0], q3[:,1], q3[:,2],'k')
    ax.plot(q4[:,0], q4[:,1], q4[:,2],'k')

    ax.plot(n1[:,0], n1[:,1], n1[:,2],'k')
    ax.plot(n2[:,0], n2[:,1], n2[:,2],'k')
    ax.plot(n3[:,0], n3[:,1], n3[:,2],'k')
    ax.plot(n4[:,0], n4[:,1], n4[:,2],'k')
    #ax.scatter(xyz[0,0], xyz[1,0], xyz[2,0], color='black')
    #ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[0,0], R[0,1], R[0,2], pivot='tail', color='red')
    #ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[1,0], R[1,1], R[1,2], pivot='tail', color='green')
    #ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[2,0], R[2,1], R[2,2], pivot='tail', color='blue')
    ax.scatter(goal[0,0], goal[1,0], goal[2,0], color='green')
    ax.set_xlabel('West/East [m]')
    ax.set_ylabel('South/North [m]')
    ax.set_zlabel('Down/Up [m]')

# generate plot points for rotors
n = 15          # number of points for plotting rotors
r = 0.1         # propeller radius. This is cosmetic only
l = 0.15        # distance from COM to COT, cosmetic only

# generate circular points
p1 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r, 0] for x in range(n+1)])
p2 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r, 0] for x in range(n+1)])
p3 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r, 0] for x in range(n+1)])
p4 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r, 0] for x in range(n+1)])

# shift to position
p1 += np.array([[l, 0., 0.] for x in range(n+1)])
p2 += np.array([[0., -l, 0.] for x in range(n+1)])
p3 += np.array([[-l, 0., 0.] for x in range(n+1)])
p4 += np.array([[0., l, 0.] for x in range(n+1)])

a1 = np.array([[l*(x/(n+1)), 0., 0.] for x in range(n+1)])
a2 = np.array([[0., -l*(x/(n+1)), 0.] for x in range(n+1)])
a3 = np.array([[-l*(x/(n+1)), 0., 0.] for x in range(n+1)])
a4 = np.array([[0., l*(x/(n+1)), 0.] for x in range(n+1)])

def main():
    # initialize filepaths
    directory = os.getcwd()
    fp = directory + "/saved_policies/"+"trpo-1-TrajectoryLine-v0.pth.tar"
    fig_path = directory + "/movies/"+args.fname

    # create list to store state information over the flight. This is... doing it the hard way,
    # but the matplotlib animation class doesn't want to do this easily :/
    state_data = []
    P = (p1, p2, p3, p4)
    A = (a1, a2, a3, a4)

    # create figure for animation function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", aspect="equal", autoscale_on=False)
    ax.set_xlim(0., 4.)
    ax.set_ylim(-2., 2.)
    ax.set_zlim(-2., 2.)
    ax.set_xlabel("West/East [m]")
    ax.set_ylabel("South/North [m]")
    ax.set_zlabel("Down/Up [m]")
    #ax.set_title(args.env + "-" + args.pol + " Trajectory Plot")

    env_name = "TrajectoryLine-v0"
    env = gym.make(env_name)
    agent = utils.load(fp)
    state = torch.Tensor(env.reset())
    goal = env.get_goal()
    done = False
    running_reward = 0
    count = 0
    position = []
    while not done:
        env.render()
        if count % 10 == 0:
            curr_pos = np.array(state[0:3]).reshape((3,1))+env.get_datum()
            curr_zeta = np.arcsin(np.array(state[3:6]).reshape((3,1)))
            position.append(curr_pos)
            plot_traj(P, A, curr_pos, curr_zeta, ax, goal)
        action  = agent.select_action(state)
        if isinstance(action, tuple):
            action = action[0]
        state, reward, done, _  = env.step(action.detach().cpu().numpy())
        state_data.append(state)
        running_reward += reward
        state = torch.Tensor(state)
        count += 1
    print("Reward: {:.3f}".format(running_reward))
    xs = [x[0,0] for x in position]
    ys = [x[1,0] for x in position]
    zs = [x[2,0] for x in position]
    ax.plot(xs, ys, zs, "r")
    ax.scatter(2., 0., 0., color='green')
    ax.scatter(3., 0., 0., color='green')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.show()
    fig.savefig(directory + "/figures/"+args.name+".pdf", bbox_inches="tight")
    
if __name__ == "__main__":
    main()