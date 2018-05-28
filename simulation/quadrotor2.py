import numpy as np
from math import sin, cos, acos, sqrt, atan2, asin

class Quadrotor:
    """
        High fidelity quadrotor simulation using quaternion rotations and a more
        robust ODE integrator. For a description of the aircraft parameters, please
        see the config file.

        -- Sean Morrison, 2018
    """
    
    def __init__(self, params):
        self.mass = params["mass"]
        self.prop_radius = params["prop_radius"]
        self.n_motors = params["n_motors"]
        self.hov_p = params["hov_p"]
        self.l = params["l"]
        self.Jxx = params["Jxx"]
        self.Jyy = params["Jyy"]
        self.Jzz = params["Jzz"]
        self.kt = params["kt"]
        self.kq = params["kq"]
        self.kd1 = params["kd1"]
        self.kd2 = params["kd2"]
        self.g = params["g"]
        self.dt = params["dt"]

        self.hov_rpm = sqrt((self.mass*self.g)/self.n_motors/self.kt)
        self.max_rpm = sqrt(1./self.hov_p)*self.hov_rpm

        self.J = np.array([[self.Jxx, 0., 0.],
                            [0., self.Jyy, 0.],
                            [0., 0., self.Jzz]])
        self.xyz = np.array([[0.],
                            [0.],
                            [0.]])
        self.zeta = np.array([[0.],
                            [0.],
                            [0.]])
        self.q = np.array([[1.],
                            [0.],
                            [0.],
                            [0.]])
        self.uvw = np.array([[0.],
                            [0.],
                            [0.]])
        self.pqr = np.array([[0.],
                            [0.],
                            [0.]])
        self.g = np.array([[0.],
                            [0.],
                            [-9.81]])
        self.rpm = np.array([0.0, 0., 0., 0.])
        
    def set_state(self, xyz, q, uvw, pqr):
        """
            Sets the state space of our vehicle
        """

        self.xyz = xyz
        self.q = q
        self. uvw = uvw
        self.pqr = pqr
    
    def get_state(self):
        """
            Returns the current state space
        """
        return self.xyz, self.q, self.uvw, self.pqr
    
    def reset(self):
        """
            Resets the initial state of the quadrotor
        """

        self.xyz = np.array([[0.],
                            [0.],
                            [0.]])
        self.q = np.array([[1.],
                            [0.],
                            [0.],
                            [0.]])
        self.uvw = np.array([[0.],
                            [0.],
                            [0.]])
        self.pqr = np.array([[0.],
                            [0.],
                            [0.]]) 
        self.rpm = np.array([0., 0., 0., 0.])
        return self.get_state()
    
    def normalize(self, v):
        return v/np.linalg.norm(v)

    def Q1(self, p):
        p0, p1, p2, p3 = p[0,0], p[1,0], p[2,0], p[3,0]
        x11 = p0**2+p1**2-p2**2-p3**2
        x12 = 2.*(p1*p2-p0*p3)
        x13 = 2.*(p1*p3+p0*p2)
        x21 = 2.*(p1*p2+p0*p3)
        x22 = p0**2-p1**2+p2**2-p3**2
        x23 = 2.*(p2*p3-p0*p1)
        x31 = 2.*(p1*p3-p0*p2)
        x32 = 2.*(p2*p3+p0*p1)
        x33 = p0**2-p1**2-p2**2+p3**2
        return np.array([[x11, x12, x13],
                        [x21, x22, x23],
                        [x31, x32, x33]])
    
    def q_mult(self, p):
        p0, p1, p2, p3 = p[0,0], p[1,0], p[2,0], p[3,0]
        return np.array([[p0, -p1, -p2, -p3],
                        [p1, p0, -p3, p2],
                        [p2, p3, p0, -p1],
                        [p3, -p2, p1, p0]])

    
    def q_conjugate(self, q):
        p0, p1, p2, p3 = q[0,0], q[1,0], q[2,0], q[3,0]
        return np.array([[p0], 
                        [-p1], 
                        [-p2], 
                        [-p3]])
    
    def q_to_euler(self, q):
        q0, q1, q2, q3 = q
        phi = atan2(2.*(q0*q1+q2*q3),q0**2-q1**2-q2**2+q3**2)
        theta = asin(2.*q0*q2-q3*q1)
        psi = atan2(2.*(q0*q3+q1*q2),q0**2+q1**2-q2**2-q3**2)
        return np.array([[phi],
                        [theta],
                        [psi]])
    
    def euler_to_q(self, zeta):
        phi, theta, psi = zeta
        q0 = cos(phi/2.)*cos(theta/2.)*cos(psi/2.)+sin(phi/2.)*sin(theta/2.)*sin(psi/2.)
        q1 = sin(phi/2.)*cos(theta/2.)*cos(psi/2.)-cos(phi/2.)*sin(theta/2.)*sin(psi/2.)
        q2 = cos(phi/2.)*sin(theta/2.)*cos(psi/2.)+sin(phi/2.)*cos(theta/2.)*sin(psi/2.)
        q3 = cos(phi/2.)*cos(theta/2.)*sin(psi/2.)-sin(phi/2.)*sin(theta/2.)*cos(psi/2.)
        return np.array([[q0],
                        [q1],
                        [q2],
                        [q3]])

    def aero_forces(self):
        """
            Calculates drag in the body xyz axis due to linear velocity
        """

        mag = np.linalg.norm(self.uvw)
        if mag == 0:
            return np.array([[0.0],[0.0],[0.0]])
        else:
            norm = self.uvw/mag
            return -(self.kd1*mag**2)*norm

    def aero_moments(self):
        """
            Models aero moments in the body xyz axis as a function of angular velocity
        """

        mag = np.linalg.norm(self.pqr)
        if mag == 0:
            return np.array([[0.0],[0.0],[0.0]])
        else:
            norm = self.pqr/mag
            return -(self.kd2*mag**2)*norm

    def thrust_forces(self, rpm):
        """
            Calculates thrust forces in the body xyz axis (E-N-U)
        """
        
        thrust = self.kt*rpm**2
        f_body_x, f_body_y = 0., 0.
        f_body_z = np.sum(thrust)
        return np.array([[f_body_x],
                        [f_body_y],
                        [f_body_z]])
    
    def thrust_moments(self, rpm):
        """
            Calculates moments about the body xyz axis due to motor thrust and torque
        """

        thrust = self.kt*rpm**2
        t_body_x = self.l*(thrust[1]-thrust[3])
        t_body_y = self.l*(thrust[2]-thrust[0])
        motor_torques = self.kq*rpm**2
        t_body_z = -motor_torques[0]+motor_torques[1]-motor_torques[2]+motor_torques[3]
        return np.array([[t_body_x],
                        [t_body_y],
                        [t_body_z]])
    
    def step(self, rpm, return_acceleration=False):
        """
            WIP
        """
        
        rpm = np.clip(rpm, 0., self.max_rpm)
        
        # thrust forces and moments, aerodynamic forces and moments
        ft = self.thrust_forces(rpm)
        mt = self.thrust_moments(rpm)
        fa = self.aero_forces()
        ma = self.aero_moments()

        # calc angular momentum
        H = self.J.dot(self.pqr)
        
        # rotate gravity vector from inertial frame to body frame
        g_b = self.Q1(self.q).dot(self.g)

        # linear and angular accelerations
        uvw_dot = (ft+fa)/self.mass+g_b-np.cross(self.pqr, self.uvw, axis=0)
        pqr_dot = np.linalg.inv(self.J).dot(((mt+ma)-np.cross(self.pqr, H, axis=0)))
        
        # forward Euler update of linear and angular velocity
        self.uvw += uvw_dot*self.dt
        self.pqr += pqr_dot*self.dt
        
        # backwards update of q_dot. We need to normalize to ensure unit quaternion
        p_pqr = np.vstack((np.array([[0]]), self.pqr))
        q_dot = -0.5*self.q_mult(self.q).dot(p_pqr)
        self.q = self.normalize(self.q+q_dot*self.dt)

        # backwards update of xyz_dot, update Euler angles
        xyz_dot = self.Q1(self.q_conjugate(self.q)).dot(self.uvw)
        self.xyz += xyz_dot*self.dt
        self.zeta = self.q_to_euler(self.q)
        if not return_acceleration:
            return self.xyz, self.q, self.uvw, self.pqr
        else:    
            return self.xyz, self.q, self.uvw, self.pqr, xyz_dot, q_dot, uvw_dot, pqr_dot