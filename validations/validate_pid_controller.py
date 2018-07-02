import numpy as np
import simulation.quadrotor as quad
import simulation.animation as ani
import simulation.config as cfg
import controller.pid_controller as ctrl
import matplotlib.pyplot as pl
from math import pi

def terminal(xyz, zeta, uvw, pqr):
    mask1 = zeta[:2] > pi/2.
    mask2 = zeta[:2] < -pi/2.
    mask3 = np.abs(xyz) > 6.
    term = np.sum(mask1+mask2)+np.sum(mask3)
    if term > 0: 
        return True
    else: 
        return False

def main():
    pl.close("all")
    pl.ion()
    fig = pl.figure(0)
    axis3d = fig.add_subplot(111, projection='3d')
    
    params = cfg.params
    iris = quad.Quadrotor(params)
    vis = ani.Visualization(iris, 10)

    goal_zeta = np.array([[0.],
                        [0.],
                        [0.]])
    goal_xyz = np.array([[3.],
                        [0.],
                        [3.]])
    xyz_init = np.array([[0.],
                        [0.],
                        [1.5]])
    uvw_init = np.array([[0.],
                        [0.],
                        [0.]])
    pqr_init = np.array([[0.],
                        [0.],
                        [0.]])

    # add some noise to the initial attitude of the aircraft
    eps = np.random.rand(3,1)/10.
    zeta_init = goal_zeta+eps
    iris.set_state(xyz_init, zeta_init, uvw_init, pqr_init)
    xyz, zeta, uvw, pqr = iris.get_state()
    
    # initialize a PD controller by setting the I term to zero
    pids = {"linear":{"p": np.array([[1.],
                                    [1.],
                                    [1.]]), 
                    "i": np.array([[0.],
                                    [0.],
                                    [0.]]), 
                    "d": np.array([[1.],
                                    [1.],
                                    [1.]])},
            "angular":{"p": np.array([[0.5],
                                    [0.5],
                                    [0.05]]), 
                    "i": np.array([[0.],
                                    [0.],
                                    [0.]]), 
                    "d": np.array([[0.2],
                                    [0.2],
                                    [0.2]])}}
    targets = {"xyz": goal_xyz,
                "zeta": goal_zeta}
    controller = ctrl.PID_Controller(iris, pids)

    counter = 0
    frames = 5
    running = True
    done = False
    t = 0.
    i = 1
    H = 5
    while running:
        states = {"xyz": xyz,
                "zeta": zeta}
        if counter%frames == 0:
            pl.figure(0)
            axis3d.cla()
            vis.draw3d(axis3d)
            vis.draw_goal(axis3d, goal_xyz.T)
            axis3d.set_xlim(-3, 3)
            axis3d.set_ylim(-3, 3)
            axis3d.set_zlim(0, 6)
            axis3d.set_xlabel('West/East [m]')
            axis3d.set_ylabel('South/North [m]')
            axis3d.set_zlabel('Down/Up [m]')
            axis3d.set_title("Time %.3f s" %t)
            pl.pause(0.001)
            pl.draw()
            pl.savefig('frame'+str(i)+".jpg")
            i += 1
        actions = controller.action(targets, states)
        xyz, zeta, uvw, pqr = iris.step(actions, rpm_commands=False)
        done = terminal(xyz, zeta, uvw, pqr)
        t += iris.dt
        #counter += 1
        if t > H:
            break 
        if done:
            print("Resetting vehicle to: {}, {}, {}, {}".format(xyz_init, zeta_init, uvw_init, pqr_init))
            iris.set_state(xyz_init, zeta_init, uvw_init, pqr_init)
            xyz, zeta, uvw, pqr = iris.get_state()
            t = 0
            counter = 0
            done = False

if __name__ == "__main__":
    main()