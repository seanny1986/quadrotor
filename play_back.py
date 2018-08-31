import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
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

# script arguments. E.g. python play_back.py --env="Hover" --pol="ppo"
parser = argparse.ArgumentParser(description="PyTorch actor-critic example")
parser.add_argument("--env", type=str, default="Hover", metavar="E", help="environment to run")
parser.add_argument("--pol", type=str, default="ppo", metavar="P", help="policy to run")
parser.add_argument("-vid", type=bool, default=True, metavar="V", help="determines whether to record video or not")
parser.add_argument("--repeats", type=int, default=3, metavar="R", help="how many attempts we want to record")
parser.add_argument("--final", type=bool, default=False, metavar="F", help="load final policy? True/False")
args = parser.parse_args()

# animation callback function
def animate(i, P, state_data, ax, g, time):

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
    goal = g[i]
    xyz, zeta, uvw, pqr = state_data[i][0:3], state_data[i][3:6], state_data[i][6:9], state_data[i][9:12]
    xyz, zeta, uvw, pqr = np.array(xyz).reshape(-1,1), np.array(zeta).reshape(-1,1), np.array(uvw).reshape(-1,1), np.array(pqr).reshape(-1,1)
    R = R1(zeta)

    # rotate to aircraft attitude
    q1 = np.einsum('ij,kj->ik', p1, R.T)
    q2 = np.einsum('ij,kj->ik', p2, R.T)
    q3 = np.einsum('ij,kj->ik', p3, R.T)
    q4 = np.einsum('ij,kj->ik', p4, R.T)

    # translate to aircraft position
    q1 = np.matlib.repmat(xyz.T,n+1,1)+q1
    q2 = np.matlib.repmat(xyz.T,n+1,1)+q2
    q3 = np.matlib.repmat(xyz.T,n+1,1)+q3
    q4 = np.matlib.repmat(xyz.T,n+1,1)+q4

    # plot rotated, translated points
    ax.cla()
    ax.plot(q1[:,0], q1[:,1], q1[:,2],'k')
    ax.plot(q2[:,0], q2[:,1], q2[:,2],'k')
    ax.plot(q3[:,0], q3[:,1], q3[:,2],'k')
    ax.plot(q4[:,0], q4[:,1], q4[:,2],'k')
    ax.scatter(xyz[0,0], xyz[1,0], xyz[2,0], color='black')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[0,0], R[0,1], R[0,2], pivot='tail', color='red')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[1,0], R[1,1], R[1,2], pivot='tail', color='green')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[2,0], R[2,1], R[2,2], pivot='tail', color='blue')
    ax.scatter(goal[0,0], goal[1,0], goal[2,0], color='green')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('West/East [m]')
    ax.set_ylabel('South/North [m]')
    ax.set_zlabel('Down/Up [m]')
    ax.set_title("Time %.3f s" %(time[i]))
    return ax

# generate plot points for rotors
n = 6           # number of points for plotting rotors
r = 0.1         # propeller radius. This is cosmetic only
l = 0.23        # distance from COM to COT, cosmetic only

# generate circular points
p1 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r, 0] for x in range(n+1)])
p2 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r, 0] for x in range(n+1)])
p3 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r, 0] for x in range(n+1)])
p4 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r, 0] for x in range(n+1)])

# shift to position
p1 += np.array([[l, 0.0, 0.0] for x in range(n+1)])
p2 += np.array([[0.0, -l, 0.0] for x in range(n+1)])
p3 += np.array([[-l, 0.0, 0.0] for x in range(n+1)])
p4 += np.array([[0.0, l, 0.0] for x in range(n+1)])

def main():
    # initialize filepaths
    directory = os.getcwd()

    fp = directory + "/saved_policies/"+args.pol+"-"+args.env+"-v0.pth.tar"
    video_path = directory + "/movies/"+args.pol+"-"+args.env
    print(fp)

    # create list to store state information over the flight. This is... doing it the hard way,
    # but the matplotlib animation class doesn't want to do this easily :/
    state_data = []
    P = (p1, p2, p3, p4)

    # create figure for animation function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", aspect="equal", autoscale_on=False)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('West/East [m]')
    ax.set_ylabel('South/North [m]')
    ax.set_zlabel('Down/Up [m]')
    ax.set_title("Time %.3f s" %(0.))

    env_name = args.env+"-v0"
    print(env_name)
    env = gym.make(env_name)
    agent = utils.load(fp)
    batch_rwd = 0
    g = []
    time = []
    for k in range(1, args.repeats+1):
        state = torch.Tensor(env.reset())
        goal = env.get_goal()
        done = False
        t = 0
        running_reward = 0
        while not done:
            g.append(goal)
            time.append(t)
            t += 0.05
            action  = agent.select_action(state)
            if isinstance(action, tuple):
                action = action[0]
            state, reward, done, _  = env.step(action.detach().cpu().numpy())
            state_data.append(state)
            running_reward += reward
            state = torch.Tensor(state)
            if done:
                break
            env.render()
        batch_rwd = (batch_rwd*(k-1)+running_reward)/k

    print("Mean reward: {:.3f}".format(batch_rwd))

    ani = animation.FuncAnimation(fig, animate, fargs=(P, state_data, ax, g, time), repeat=False, frames=len(state_data), interval=50)
    #plt.show()

    print("Saving video in: "+video_path+".mp4")

    #ani.save(video_path+".mp4", writer=FFwriter)#, extra_args=['-loglevel', 'verbose'])
    #plt.show()

if __name__ == "__main__":
    main()
