import numpy as np
from math import sin, cos

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