import numpy as np
import simulation.quadrotor as quad
import simulation.animation as ani
import simulation.config as cfg
import matplotlib.pyplot as pl
import scipy.optimize as opt
from math import pi

class PID_Controller:
    def __init__(self, aircraft, p_gain, i_gain, d_gain):
        self.aircraft = aircraft
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.last_error = 0.
        self.i_error = 0.
        
        self.kt = aircraft.kt
        self.kq = aircraft.kq
        self.mass = aircraft.mass
        self.J = aircraft.J
        self.g = aircraft.g
        self.dt = aircraft.dt
        self.hov_rpm = aircraft.hov_rpm
        self.min_rpm = 0.
        self.max_rpm = aircraft.max_rpm
    
    def compute(self, state, target):
        error = target-state
        p_error = error
        self.i_error += (error + self.last_error)*self.dt
        i_error = self.i_error
        d_error = (error-self.last_error)/self.dt
        p_output = self.p_gain*p_error
        i_output = self.i_gain*i_error
        d_output = self.d_gain*d_error
        self.last_error = error
        return p_output+i_output+d_output
    
    def action(self, state, target):
        u = self.compute(state, target)
        req_torques = -self.J.dot(u)
        print(req_torques)
        input("Press any key")
        bnds = ((self.min_rpm, self.max_rpm),
                (self.min_rpm, self.max_rpm),
                (self.min_rpm, self.max_rpm),
                (self.min_rpm, self.max_rpm))
        x0 = np.array([self.hov_rpm, self.hov_rpm, self.hov_rpm, self.hov_rpm])
        rpm = opt.minimize(self.cost, x0, args=(req_torques, state),method='L-BFGS-B', bounds=bnds)
        print(rpm)
        return rpm.x
    
    def cost(self, rpm, req_torques, state):
        torques = self.aircraft.thrust_torques(rpm)
        thrust = self.aircraft.thrust_forces(rpm)
        thrust_i = self.aircraft.R1(state).dot(thrust)
        mg = self.mass*self.g
        torque_cost = np.mean((req_torques-torques)**2)
        hover_cost = (thrust_i[2,0]-mg[2,0])**2
        return torque_cost+hover_cost

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
    controller = PID_Controller(iris, 0.5, 0.5, 0.5)

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
        rpm = controller.action(zeta, goal_zeta)
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