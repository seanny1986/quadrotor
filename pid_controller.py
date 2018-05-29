import numpy as np
import simulation.quadrotor as quad
import simulation.animation as ani
import simulation.config as cfg
import matplotlib.pyplot as pl
import scipy.optimize as opt
from math import pi

class PID_Controller:
    def __init__(self, aircraft, gains):
        self.aircraft = aircraft
        self.p_lin = gains.linear.p
        self.i_lin = gains.linear.i
        self.d_lin = gains.linear.d
        self.p_ang = gains.angular.p
        self.i_ang = gains.angular.i
        self.d_ang = gains.angular.d
        self.last_lin_error = 0.
        self.last_ang_error = 0.
        self.lin_i_error = 0.
        self.ang_i_error = 0.
        
        self.kt = aircraft.kt
        self.kq = aircraft.kq
        self.mass = aircraft.mass
        self.J = aircraft.J
        self.g = aircraft.g
        self.dt = aircraft.dt
        self.hov_rpm = aircraft.hov_rpm
        self.min_rpm = 0.
        self.max_rpm = aircraft.max_rpm
    
    def compute_lin_pid(self, state, target):
        error = target-state
        p_error = error
        self.lin_i_error += (error + self.last_lin_error)*self.dt
        d_error = (error-self.last_lin_error)/self.dt
        p_output = self.p_lin*p_error
        i_output = self.i_lin*self.lin_i_error
        d_output = self.d_lin*d_error
        self.last_lin_error = error
        return p_output+i_output+d_output
    
    def compute_ang_pid(self, state, target, lin_pid):
        error = target-state
        p_error = error
        self.ang_i_error += (error + self.last_ang_error)*self.dt
        d_error = (error-self.last_ang_error)/self.dt
        p_output = self.p_ang*p_error
        i_output = self.i_ang*self.ang_i_error
        d_output = self.d_ang*d_error
        self.last_ang_error = error
        return p_output+i_output+d_output
    
    def action(self, state, target):
        u1 = -self.compute_lin_pid(state.xyz, target.xyz)
        u2 = -self.compute_ang_pid(state.zeta, target.zeta, u1)
        throttle = u1[2,0]
        o1 = throttle+u2[0,0]-u2[2,0]
        o2 = throttle+u2[1,0]+u2[2,0]
        o3 = throttle-u2[0,0]-u2[2,0]
        o4 = throttle-u2[1,0]+u2[2,0]
        return np.array([o1, o2, o3, o4])

def terminal(xyz, zeta, uvw, pqr):
    mask1 = zeta > pi/2.
    mask2 = zeta < -pi/2.
    mask3 = np.abs(xyz) > 6.
    term = np.sum(mask1+mask2+mask3)
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
    hover_rpm = iris.hov_rpm
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    vis = ani.Visualization(iris, 10)
    rpm = trim+50

    goal_zeta = np.array([[0.],
                        [0.],
                        [0.]])
    goal_xyz = np.array([[0.],
                        [0.],
                        [2.]])
    xyz_init = np.array([[0.],
                        [0.],
                        [1.5]])
    uvw_init = np.array([[0.],
                        [0.],
                        [0.]])
    pqr_init = np.array([[0.],
                        [0.],
                        [0.]])

    eps = np.random.rand(3,1)
    zeta_init = goal_zeta+eps
    iris.set_state(xyz_init, zeta_init, uvw_init, pqr_init)
    xyz, zeta, uvw, pqr = iris.get_state()
    gains = {"linear":{"p": 1.,
                        "i": 1.,
                        "d": 1.},
            "angular":{"p": 1.,
                        "i": 1.,
                        "d": 1.}}
    targets = {"xyz": goal_xyz,
                "zeta": goal_zeta}
    controller = PID_Controller(iris, gains)

    counter = 0
    frames = 100
    running = True
    done = False
    t = 0
    
    while running:
        if counter%frames == 0:
            pl.figure(0)
            axis3d.cla()
            vis.draw3d(axis3d)
            axis3d.set_xlim(-3, 3)
            axis3d.set_ylim(-3, 3)
            axis3d.set_zlim(0, 6)
            axis3d.set_xlabel('West/East [m]')
            axis3d.set_ylabel('South/North [m]')
            axis3d.set_zlabel('Down/Up [m]')
            axis3d.set_title("Time %.3f s" %t)
            pl.pause(0.001)
            pl.draw()
        states = {"xyz": xyz,
                "zeta": zeta}
        rpm = controller.action(states, targets)
        xyz, zeta, uvw, pqr = iris.step(rpm)
        done = terminal(xyz, zeta, uvw, pqr)
        t += iris.dt
        if done:
            print("Resetting vehicle")
            iris.set_state(xyz_init, zeta_init, uvw_init, pqr_init)
            xyz, zeta, uvw, pqr = iris.get_state()
            t = 0
            counter = 0
            done = False

if __name__ == "__main__":
    main()