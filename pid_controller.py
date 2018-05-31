import numpy as np
import simulation.quadrotor as quad
import simulation.animation as ani
import simulation.config as cfg
import matplotlib.pyplot as pl
import scipy.optimize as opt
import copy
from math import pi

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
        
        self.kt = aircraft.kt
        self.kq = aircraft.kq
        self.mass = aircraft.mass
        self.J = aircraft.J
        self.g = aircraft.g
        self.dt = aircraft.dt
        self.hov_rpm = aircraft.hov_rpm
        self.max_rpm = aircraft.max_rpm
        self.lin_bnd = ((-15., 15),
                        (-15., 15),
                        (-15., 15),
                        (-15., 15))
        self.ang_bnd = ((-2.5, 2.5),
                        (-2.5, 2.5),
                        (-2.5, 2.5),
                        (-2.5, 2.5))
        self.weight = self.mass*self.aircraft.G
    
    def compute_lin(self, state, target):
        error = target-state
        p_error = error
        self.i_error_xyz += (error + self.last_error_xyz)*self.dt
        i_error = self.i_error_xyz
        d_error = (error-self.last_error_xyz)/self.dt
        p_output = self.p_xyz*p_error
        i_output = self.i_xyz*i_error
        d_output = self.d_xyz*d_error
        self.last_error_xyz = error
        print(p_output+i_output+d_output)
        return p_output+i_output+d_output
    
    def compute_ang(self, state, target):
        error = target-state
        p_error = error
        self.i_error_zeta += (error + self.last_error_zeta)*self.dt
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
        forces = self.compute_lin(xyz, target_xyz)
        moments = -self.compute_ang(zeta, target_zeta)#+forces*np.array([[1.],
                                                                        #[1.],
                                                                        #[0.]])
        z = forces*np.array([[0.],
                            [0.],
                            [1.]])
        x0 = np.array([0., 0., 0., 0.])
        rpm_f = opt.minimize(self.force_cost, x0, args=(z, zeta), method='L-BFGS-B', bounds=self.lin_bnd)
        rpm_m = opt.minimize(self.moment_cost, x0, args=(moments, zeta), method='L-BFGS-B', bounds=self.ang_bnd)
        print(rpm_f.x)
        print(rpm_m.x)
        print(self.hov_rpm+rpm_f.x+rpm_m.x)
        input("Pause")
        return self.hov_rpm+rpm_f.x+rpm_m.x
    
    def force_cost(self, rpm, req_forces, zeta):
        print("req forces:")
        print(req_forces)
        forces = self.aircraft.thrust_forces(self.hov_rpm+rpm)
        mapped = self.aircraft.R1(zeta).dot(forces)*np.array([[0.],[0.],[1.]])
        print("mapped:")
        mapped += self.weight
        print(mapped)
        force_cost = -0.5*(req_forces-mapped)**2
        print(force_cost)
        return np.sum(force_cost)

    def moment_cost(self, rpm, req_moments, zeta):
        moments = self.aircraft.thrust_moments(self.hov_rpm+rpm)
        mapped = np.linalg.inv(self.aircraft.R2(zeta)).dot(moments)
        moment_cost = -0.5*(req_moments-mapped)**2
        return np.sum(moment_cost)

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
    rpm = trim

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
                    "i": np.array([[1.],
                                    [1.],
                                    [1.]]), 
                    "d": np.array([[1.],
                                    [1.],
                                    [1.]])},
            "angular":{"p": np.array([[1.],
                                    [1.],
                                    [1.]]), 
                    "i": np.array([[1.],
                                    [1.],
                                    [1.]]), 
                    "d": np.array([[1.],
                                    [1.],
                                    [1.]])}}
    targets = {"xyz": goal_xyz,
                "zeta": goal_zeta}
    controller = PID_Controller(iris, pids)

    counter = 0
    frames = 100
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
            axis3d.set_xlim(-3, 3)
            axis3d.set_ylim(-3, 3)
            axis3d.set_zlim(0, 6)
            axis3d.set_xlabel('West/East [m]')
            axis3d.set_ylabel('South/North [m]')
            axis3d.set_zlabel('Down/Up [m]')
            axis3d.set_title("Time %.3f s" %t)
            pl.pause(0.001)
            pl.draw()
        rpm = controller.action(states, targets)
        xyz, zeta, uvw, pqr = iris.step(rpm)
        done = terminal(xyz, zeta, uvw, pqr)
        t += iris.dt
        if done:
            print("Resetting vehicle to: {}, {}, {}, {}".format(xyz_init, zeta_init, uvw_init, pqr_init))
            iris.set_state(xyz_init, zeta_init, uvw_init, pqr_init)
            xyz, zeta, uvw, pqr = iris.get_state()
            print(xyz)
            t = 0
            counter = 0
            done = False

if __name__ == "__main__":
    main()