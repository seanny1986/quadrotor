import torch
import simulation.quadrotor as quad
import simulation.animation as ani
import simulation.config as cfg
import matplotlib.pyplot as pl
from math import pi, sqrt
import numpy as np

def main():
    pl.close("all")
    pl.ion()
    fig = pl.figure(0)
    axis3d = fig.add_subplot(111, projection='3d')
    
    params = cfg.params
    mass = params["mass"]
    kt = params["kt"]
    dt = params["dt"]
    T = 15

    time = np.linspace(0, T, T/dt)

    hover_thrust = (mass*9.81)/4.0
    hover_rpm = sqrt(hover_thrust/kt)
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    iris = quad.Quadrotor(params)
    vis = ani.Visualization(iris, 10)

    counter = 0
    frames = 100
    rpm = trim+50

    policy = torch.load("ddpg_policy.pth.tar")
    for t in time:
        xyz, zeta, uvw, pqr = iris.get_state()
        zeta = torch.from_numpy(zeta.T).float()
        uvw = torch.from_numpy(uvw.T).float()
        pqr = torch.from_numpy(pqr.T).float()
        state = torch.cat([zeta.sin(), zeta.cos(), uvw, pqr],dim=1)
        rpm = policy.select_action(state)
        iris.step(rpm.numpy()[0])
        
        # Animation
        if counter%frames == 0:

            pl.figure(0)
            axis3d.cla()
            vis.draw3d(axis3d, iris.xyz, iris.R1(iris.zeta).T)
            axis3d.set_xlim(-3, 3)
            axis3d.set_ylim(-3, 3)
            axis3d.set_zlim(0, 6)
            axis3d.set_xlabel('East/West [m]')
            axis3d.set_ylabel('North/South [m]')
            axis3d.set_zlabel('Up/Down [m]')
            axis3d.set_title("Time %.3f s" %t)
            pl.pause(0.001)
            pl.draw()
        
        mask = zeta.numpy() > 30*pi/180
        if np.sum(mask) > 0:
            #print("Breaking loop at time {}".format(t*iris.dt))
            break

if __name__ == "__main__":
    main()