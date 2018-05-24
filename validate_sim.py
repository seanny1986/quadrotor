import simulation.quadrotor as quad
import math
import numpy as np
import matplotlib.pyplot as pl
import simulation.animation as ani

def main():
    pl.close("all")
    pl.ion()
    fig = pl.figure(0)
    axis3d = fig.add_subplot(111, projection='3d')
    
    mass = 0.65
    prop_radius = 0.1
    l = 0.23
    Jxx = 7.5e-3
    Jyy = 7.5e-3
    Jzz = 1.3e-2
    kt = 3.13e-5
    kq = 7.5e-7
    kd1 = 9e-3
    kd2 = 9e-4
    dt = 0.05
    T = 1.5

    time = np.linspace(0, T, T/dt)

    hover_thrust = (mass*9.81)/4.0
    hover_rpm = math.sqrt(hover_thrust/kt)
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    iris = quad.Quadrotor(mass, prop_radius, l, Jxx, Jyy, Jzz, kt, kq, kd1, kd2, dt)
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
            vis.draw3d(axis3d, iris.xyz, iris.R1(iris.zeta).T)
            axis3d.set_xlim(-5, 5)
            axis3d.set_ylim(-5, 5)
            axis3d.set_zlim(0, 10)
            axis3d.set_xlabel('East/West [m]')
            axis3d.set_ylabel('North/South [m]')
            axis3d.set_zlabel('Up/Down [m]')
            axis3d.set_title("Time %.3f s" %t)
            pl.pause(0.001)
            pl.draw()
        
        rpm += np.array([0.5,0.0,0.0,0.0])

if __name__ == "__main__":
    main()