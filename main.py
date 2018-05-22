import quad
import math
import numpy as np
import matplotlib.pyplot as pl
import animation as ani

def main():

    quadcolor = ['k']
    pl.close("all")
    pl.ion()
    fig = pl.figure(0)
    axis3d = fig.add_subplot(111, projection='3d')
    
    mass = 0.65
    l = 0.23
    Jxx = 7.5e-3
    Jyy = 7.5e-3
    Jzz = 1.3e-2
    kt = 3.13e-5
    kq = 7.5e-7
    kd1 = 9e-3
    kd2 = 9e-4
    dt = 0.01
    T = 5

    time = np.linspace(0, T, T/dt)

    hover_thrust = (mass*9.81)/4.0
    hover_rpm = math.sqrt(hover_thrust/kt)
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    iris = quad.Quadrotor(mass, l, Jxx, Jyy, Jzz, kt, kq, kd1, kd2, dt)

    counter = 0
    frames = 100
    for t in time:
        rpm = trim+50

        iris.step(rpm)
        
        # Animation
        if counter%frames == 0:

            pl.figure(0)
            axis3d.cla()
            ani.draw3d(axis3d, iris.xyz, iris.R1(iris.zeta), quadcolor[0])
            axis3d.set_xlim(-10, 10)
            axis3d.set_ylim(-10, 10)
            axis3d.set_zlim(0, 15)
            axis3d.set_xlabel('South [m]')
            axis3d.set_ylabel('East [m]')
            axis3d.set_zlabel('Up [m]')
            axis3d.set_title("Time %.3f s" %t)
            pl.pause(0.001)
            pl.draw()

if __name__ == "__main__":
    main()