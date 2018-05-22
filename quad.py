import numpy as np
from math import sin, cos, tan

class Quadrotor():
    def __init__(self, mass, l, Jxx, Jyy, Jzz, kt, kq, kd1, kd2, dt):
        self.mass = mass
        self.J = np.array([[Jxx,0.0,0.0],
                            [0.0,Jyy,0.0],
                            [0.0,0.0,Jzz]])
        self.kt = kt
        self.kq = kq
        self.kd1 = kd1
        self.kd2 = kd2
        self.l = l
        self.xyz = np.array([[0.0],
                            [0.0],
                            [0.0]])
        self.zeta = np.array([[0.0],
                            [0.0],
                            [0.0]])
        self.uvw = np.array([[0.0],
                            [0.0],
                            [0.0]])
        self.pqr = np.array([[0.0],
                            [0.0],
                            [0.0]])
        self.g = np.array([[0.0],
                            [0.0],
                            [9.81]])
        self.rpm = np.array([0.0,0.0,0.0,0.0])
        self.dt = dt

    def set_state(self, xyz, zeta, uvw, pqr):
        self.xyz = xyz
        self.zeta = zeta
        self. uvw = uvw
        self.pqr = pqr
    
    def get_state(self):
        return self.xyz, self.zeta, self.uvw, self.pqr

    def R1(self, zeta):
        phi = zeta[0,0]
        theta = zeta[1,0]
        psi = zeta[2,0]
        x11 = cos(theta)*cos(psi)
        x12 = -cos(theta)*sin(psi)+sin(phi)*sin(theta)*cos(psi)
        x13 = sin(phi)*sin(psi)+cos(phi)*sin(theta)*cos(psi)
        x21 = cos(theta)*sin(psi)
        x22 = cos(phi)*cos(psi)+sin(phi)*sin(theta)*sin(psi)
        x23 = -sin(phi)*cos(psi)+cos(phi)*sin(theta)*sin(psi)
        x31 = -sin(theta)
        x32 = sin(phi)*cos(theta)
        x33 = cos(phi)*cos(theta)
        return np.array([[x11, x12, x13],
                        [x21, x22, x23],
                        [x31, x32, x33]])

    def R2(self, zeta):
        phi = zeta[0,0]
        theta = zeta[1,0]
        x11 = 1
        x12 = sin(phi)*tan(theta)
        x13 = cos(phi)*tan(theta)
        x21 = 0
        x22 = cos(phi)
        x23 = -sin(phi)
        x31 = 0
        x32 = sin(phi)/cos(theta)
        x33 = cos(phi)/cos(theta)
        return np.array([[x11, x12, x13],
                        [x21, x22, x23],
                        [x31, x32, x33]])

    def motor_thrusts(self, rpm):
        return self.kt*rpm**2
    
    def aero_forces(self):
        mag = np.linalg.norm(self.uvw)
        if mag == 0:
            return np.array([[0.0],[0.0],[0.0]])
        else:
            norm = self.uvw/mag
            return -(self.kd1*mag**2)*norm

    def aero_moments(self):
        mag = np.linalg.norm(self.pqr)
        if mag == 0:
            return np.array([[0.0],[0.0],[0.0]])
        else:
            norm = self.pqr/mag
            return -(self.kd2*mag**2)*norm

    def thrust_forces(self, thrust):
        f_body_x, f_body_y = 0, 0
        f_body_z = np.sum(thrust)
        return np.array([[f_body_x],
                        [f_body_y],
                        [f_body_z]])
    
    def thrust_torques(self, thrust, rpm):
        t_body_x = self.l*(thrust[1]-thrust[3])
        t_body_y = self.l*(thrust[2]-thrust[0])
        motor_torques = self.kq*rpm**2
        t_body_z = -motor_torques[0]+motor_torques[1]-motor_torques[2]+motor_torques[3]
        return np.array([[t_body_x],
                        [t_body_y],
                        [t_body_z]])
    
    def step(self, rpm):
        r1 = self.R1(self.zeta)
        r2 = self.R2(self.zeta)
        thrust = self.motor_thrusts(rpm)
        fm = self.thrust_forces(thrust)
        tm = self.thrust_torques(thrust, rpm)
        fa = self.aero_forces()
        ta = self.aero_moments()
        Jw = self.J.dot(self.pqr)
        pqr_dot = np.linalg.inv(self.J).dot(((tm+ta)-np.cross(self.pqr,Jw,axis=0)))
        uvw_dot = (fm+fa)/self.mass-np.linalg.inv(r1).dot(self.g)
        self.uvw += uvw_dot*self.dt
        self.pqr += pqr_dot*self.dt
        xyz_dot = r1.dot(self.uvw)
        zeta_dot = r2.dot(self.pqr)
        self.xyz += xyz_dot*self.dt
        self.zeta += zeta_dot*self.dt
        return self.xyz, self.zeta, self.uvw, self.pqr, xyz_dot, zeta_dot, uvw_dot, pqr_dot


