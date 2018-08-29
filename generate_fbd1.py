import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import gym
import gym_aero
import numpy as np
import numpy.matlib
import utils
import argparse
import config as cfg
from math import sin, cos, tan, pi
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

"""
    Function to play back saved policies and save video. I hate matplotlib.

    -- Sean Morrison, 2018
"""

# script arguments. E.g. python play_back.py --env="Hover" --pol="ppo"
parser = argparse.ArgumentParser(description="PyTorch actor-critic example")
parser.add_argument('--name', type=str, default='fig', metavar='N', help='name to save figure as')
args = parser.parse_args()

def lims(mplotlims):
    scale = 1.021
    offset = (mplotlims[1] - mplotlims[0])*scale
    return mplotlims[1] - offset, mplotlims[0] + offset

# animation callback function
def plot_ac(P, A, state, ax):
    
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
    xyz, zeta = state
    xyz, zeta = xyz.reshape(-1,1), zeta.reshape(-1,1)
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

    ax.scatter(xyz[0,0], xyz[1,0], xyz[2,0], color='black')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], 0.5*R[0,0], 0.5*R[0,1], 0.5*R[0,2], pivot='tail', color='black')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], 0.5*R[1,0], 0.5*R[1,1], 0.5*R[1,2], pivot='tail', color='black')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], 0.5*R[2,0], 0.5*R[2,1], 0.5*R[2,2], pivot='tail', color='black')
    
    ax.text(0.5*R[0,0]+xyz[0,0]+0.02, 0.5*R[0,1]+xyz[1,0]+0.02, 0.5*R[0,2]+xyz[2,0]+0.02, r"$X_{b}$", None)
    ax.text(0.5*R[1,0]+xyz[0,0]+0.02, 0.5*R[1,1]+xyz[1,0]+0.02, 0.5*R[1,2]+xyz[2,0]+0.02, r"$Y_{b}$", None)
    ax.text(0.5*R[2,0]+xyz[0,0]+0.02, 0.5*R[2,1]+xyz[1,0]+0.02, 0.5*R[2,2]+xyz[2,0]+0.02, r"$Z_{b}$", None)
    
    ax.set_xlim(-0.5, 1.)
    ax.set_ylim(-0.5, 1.)
    ax.set_zlim(-0.5, 1.)
    #ax.set_xlabel('West/East [m]')
    #ax.set_ylabel('South/North [m]')
    #ax.set_zlabel('Down/Up [m]')

# generate plot points for rotors
n = 15          # number of points for plotting rotors
r = 0.1         # propeller radius. This is cosmetic only
l = 0.15        # distance from COM to COT, cosmetic only

xyz = np.array([[0.5],[0.8],[0.25]])
zeta = np.array([[-pi/8.],[pi/8.],[pi/8.]])

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

a1 = np.array([[(l-r)*(x/(n+1)), 0., 0.] for x in range(n+1)])
a2 = np.array([[0., -(l-r)*(x/(n+1)), 0.] for x in range(n+1)])
a3 = np.array([[-(l-r)*(x/(n+1)), 0., 0.] for x in range(n+1)])
a4 = np.array([[0., (l-r)*(x/(n+1)), 0.] for x in range(n+1)])

def main():
    # initialize filepaths
    directory = os.getcwd()

    # create list to store state information over the flight. This is... doing it the hard way,
    # but the matplotlib animation class doesn't want to do this easily :/
    P = (p1, p2, p3, p4)
    A = (a1, a2, a3, a4)

    # create figure for animation function
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", aspect="equal", autoscale_on=False)
    plot_ac(P, A, (xyz, zeta), ax)
    ax.quiver(0., 0., 0., 0.5, 0., 0., pivot='tail', color='black')
    ax.quiver(0., 0., 0., 0., 0.5, 0., pivot='tail', color='black')
    ax.quiver(0., 0., 0., 0., 0., 0.5, pivot='tail', color='black')
    ax.text(0.55, 0., 0., r"$X_{i}$", None)
    ax.text(0., 0.55, 0., r"$Y_{i}$", None)
    ax.text(0., 0., 0.55, r"$Z_{i}$", None)
    dat = np.hstack([np.array([[0.],[0.],[0.]]), xyz])
    ax.plot(dat[0,:], dat[1,:], dat[2,:], "k", ls="--", lw=1.)

    ax.set_xlim(-0.5, 1.)
    ax.set_ylim(-0.5, 1.)
    ax.set_zlim(-0.5, 1.)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.set_alpha(0.5)
    ax.yaxis.pane.set_alpha(0.5)
    ax.zaxis.pane.set_alpha(0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    #ax.view_init(elev=25, azim=130-90-30)
    xlims, ylims, zlims = lims(ax.get_xlim()), lims(ax.get_ylim()), lims(ax.get_zlim())
    i = np.array([xlims[0], ylims[0], zlims[0]])
    f = np.array([xlims[0], ylims[0], zlims[1]])
    p = art3d.Poly3DCollection(np.array([[i, f]]))
    p.set_color('black')
    ax.add_collection3d(p)
    fig.tight_layout()
    plt.show()
    fig.savefig(directory + "/figures/"+args.name+".pdf", bbox_inches="tight")
    
if __name__ == "__main__":
    main()