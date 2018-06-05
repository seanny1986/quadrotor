import numpy as np
from math import sin, cos

"""
    Implements a linearized hover pid controller for a quadrotor. We want to solve the equation:

    [[u1],      [[kt,   kt,     kt,     kt],    [[w_1^2], 
    [u2],   =   [0.,    lkt,    0.,   -lkt],     [w_2^2],
    [u3],       [-lkt,  0.,     lkt,    0.],     [w_3^2],
    [u4]]       [-kq,   kq,    -kq,     kq]]     [w_4^2]]

    Where u1 is our thrust in the body z-direction, u2 is the commanded roll, u3 is the commanded
    pitch, and u4 is the commanded yaw. To do this, we calculate the PID error in xyz, which gives
    us a force vector in the body frame. Next, we rotate the x and y components of this vector about 
    the z-axis to get our roll and pitch commands. We feed this into the zeta PID controller as our 
    target, along with a desired yaw of 0. The returned PID error is our U2 vector, where:

    U2 = [u2, u3, u4]^T

    We stack u1 on top of this vector to get [u1, u2, u3, u4]^T, and solve the above equation to
    get the RPM values.
"""

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

        self.mass = aircraft.mass
        self.g = aircraft.g
        self.dt = aircraft.dt
        
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
    
    def action(self, target, state):
        xyz = state["xyz"]
        zeta = state["zeta"]
        target_xyz = target["xyz"]
        target_zeta = target["zeta"]
        u_s = self.compute_lin_pid(target_xyz, xyz)
        u_1 = np.array([[self.mass*(self.g+u_s[2,0])]])
        phi_c = 1./self.g*(u_s[0,0]*sin(target_zeta[2,0])-u_s[1,0]*cos(target_zeta[2,0]))
        theta_c = 1./self.g*(u_s[0,0]*cos(target_zeta[2,0])+u_s[1,0]*sin(target_zeta[2,0]))
        psi_c = 0.
        zeta_c = np.array([[phi_c],
                        [theta_c],
                        [psi_c]])
        u_2 = self.compute_ang_pid(zeta_c, zeta)*np.array([[1.],[1.],[-1.]])
        return np.vstack((u_1, u_2))