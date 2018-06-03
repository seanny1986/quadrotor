import numpy as np
import simulation.quadrotor as quad
import simulation.animation as ani
import simulation.config as cfg
import matplotlib.pyplot as pl
from math import pi, sin, cos

class PID_Controller:
    def __init__(self, aircraft, pids):
        self.aircraft = aircraft
        self.p_xyz = pids["linear"]["p"]
        self.i_xyz = pids["linear"]["i"]
        self.d_xyz = pids["linear"]["d"]
        self.p_zeta = pids["angular"]["p"]
        self.i_zeta = pids["angular"]["i"]
        self.d_zeta = pids["angular"]["d"]
        self.last_error_xyz = 0.
        self.i_error_xyz = 0.
        self.last_error_zeta = 0.
        self.i_error_zeta = 0.
        self.i_limit_xyz = 50.
        self.i_limit_zeta = 50.

        self.kt = aircraft.kt
        self.kq = aircraft.kq
        self.mass = aircraft.mass
        self.J = aircraft.J
        self.g = aircraft.g
        self.dt = aircraft.dt
        self.hov_rpm = aircraft.hov_rpm
        self.max_rpm = aircraft.max_rpm
        self.n_motors = aircraft.n_motors
        
    def compute_lin_pid(self, target, state):
        error = target-state
        p_error = error
        self.i_error_xyz += (error + self.last_error_xyz)*self.dt
        self.i_error_xyz = np.clip(self.i_error_xyz, 0., self.i_limit_xyz)
        i_error = self.i_error_xyz
        d_error = (error-self.last_error_xyz)/self.dt
        p_output = self.p_xyz*p_error
        i_output = self.i_xyz*i_error
        d_output = self.d_xyz*d_error
        self.last_error_xyz = error
        return p_output+i_output+d_output
    
    def compute_ang_pid(self, target, state):
        error = target-state
        p_error = error
        self.i_error_zeta += (error + self.last_error_zeta)*self.dt
        self.i_error_zeta = np.clip(self.i_error_zeta, 0., self.i_limit_zeta)
        i_error = self.i_error_zeta
        d_error = (error-self.last_error_zeta)/self.dt
        p_output = self.p_zeta*p_error
        i_output = self.i_zeta*i_error
        d_output = self.d_zeta*d_error
        self.last_error_zeta = error
        return p_output+i_output+d_output
    
    def action(self, state, target):
        xyz = state["xyz"]
        zeta = state["zeta"]
        target_xyz = target["xyz"]
        target_zeta = target["zeta"]
        u_s = self.compute_lin_pid(target_xyz, xyz)
        u_1 = np.array([[self.mass*(self.g+u_s[2,0])]])
        phi_c = 1./self.g*(u_s[0,0]*sin(target_zeta[2,0])-u_s[1,0]*cos(target_zeta[2,0]))
        theta_c = 1./self.g*(u_s[0,0]*cos(target_zeta[2,0])+u_s[1,0]*sin(target_zeta[2,0]))
        psi_c = 0.
        angs = np.array([[phi_c],
                        [theta_c],
                        [psi_c]])
        u_2 = self.compute_ang_pid(angs, zeta)
        return np.vstack((u_1, u_2))

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
    goal_xyz = np.array([[0.],
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

    eps = np.random.rand(3,1)/10.
    zeta_init = goal_zeta+eps
    iris.set_state(xyz_init, zeta_init, uvw_init, pqr_init)
    xyz, zeta, uvw, pqr = iris.get_state()
    
    pids = {"linear":{"p": np.array([[1.],
                                    [1.],
                                    [1.]]), 
                    "i": np.array([[0.01],
                                    [0.01],
                                    [0.01]]), 
                    "d": np.array([[1.],
                                    [1.],
                                    [1.]])},
            "angular":{"p": np.array([[0.1],
                                    [0.1],
                                    [0.01]]), 
                    "i": np.array([[0.001],
                                    [0.001],
                                    [0.001]]), 
                    "d": np.array([[0.1],
                                    [0.1],
                                    [0.1]])}}
    targets = {"xyz": goal_xyz,
                "zeta": goal_zeta}
    controller = PID_Controller(iris, pids)

    counter = 0
    frames = 5
    running = True
    done = False
    t = 0
    
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
        actions = controller.action(states, targets)
        xyz, zeta, uvw, pqr = iris.step(actions, rpm_commands=False)
        done = terminal(xyz, zeta, uvw, pqr)
        t += iris.dt
        #counter += 1
        if done:
            print("Resetting vehicle to: {}, {}, {}, {}".format(xyz_init, zeta_init, uvw_init, pqr_init))
            iris.set_state(xyz_init, zeta_init, uvw_init, pqr_init)
            xyz, zeta, uvw, pqr = iris.get_state()
            t = 0
            counter = 0
            done = False

if __name__ == "__main__":
    main()