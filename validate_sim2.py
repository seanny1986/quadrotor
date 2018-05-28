import simulation.quadrotor2 as quad
import simulation.animation as ani
import simulation.config as cfg
import numpy as np
import matplotlib.pyplot as pl

def main():
    pl.close("all")
    pl.ion()
    fig = pl.figure(0)
    axis3d = fig.add_subplot(111, projection='3d')
    
    params = cfg.params
    iris = quad.Quadrotor(params)
    T = 3
    time = np.linspace(0, T, T/iris.dt)
    hover_rpm = iris.hov_rpm
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    vis = ani.Visualization(iris, 10, quaternion=True)

    counter = 0
    frames = 100
    rpm = trim+50

    for t in time:
        iris.step(rpm)
        if counter%frames == 0:
            pl.figure(0)
            axis3d.cla()
            vis.draw3d_quat(axis3d)
            axis3d.set_xlim(-3, 3)
            axis3d.set_ylim(-3, 3)
            axis3d.set_zlim(0, 6)
            axis3d.set_xlabel('West/East [m]')
            axis3d.set_ylabel('South/North [m]')
            axis3d.set_zlabel('Down/Up [m]')
            axis3d.set_title("Time %.3f s" %t)
            pl.pause(0.001)
            pl.draw()
        rpm += np.array([0., 0., 0., 0.25])

if __name__ == "__main__":
    main()