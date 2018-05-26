import simulation.quadrotor as quad
import simulation.animation as ani
import simulation.config as cfg
import math
import numpy as np
import matplotlib.pyplot as pl

def main():
    pl.close("all")
    pl.ion()
    fig = pl.figure(0)
    axis3d = fig.add_subplot(111, projection='3d')
    
    params = cfg.params
    mass = params["mass"]
    kt = params["kt"]
    dt = params["dt"]
    T = 2.5

    time = np.linspace(0, T, T/dt)

    hover_thrust = (mass*9.81)/4.0
    hover_rpm = math.sqrt(hover_thrust/kt)
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    iris = quad.Quadrotor(params)
    vis = ani.Visualization(iris, 10)

    counter = 0
    frames = 100
    rpm = trim+50

    for t in time:

        iris.step(rpm)
        
        # Animation
        if counter%frames == 0:

            pl.figure(0)
            axis3d.cla()
            vis.draw3d(axis3d)
            axis3d.set_xlim(-3, 3)
            axis3d.set_ylim(-3, 3)
            axis3d.set_zlim(0, 6)
            axis3d.set_xlabel('East/West [m]')
            axis3d.set_ylabel('North/South [m]')
            axis3d.set_zlabel('Up/Down [m]')
            axis3d.set_title("Time %.3f s" %t)
            pl.pause(0.001)
            pl.draw()
        rpm += np.array([0.5, 0., 0., 0.])

if __name__ == "__main__":
    main()