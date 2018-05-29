import numpy as np
from math import sin, cos, acos, sqrt, atan2, asin

class Quadrotor:
    """
        Higher fidelity quadrotor simulation using quaternion rotations and a second
        order ODE integrator. For a description of the aircraft parameters, please
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
        self.kd = params["kd"]
        self.km = params["km"]
        self.g = params["g"]
        self.dt = params["dt"]

        self.J = np.array([[self.Jxx, 0., 0.],
                            [0., self.Jyy, 0.],
                            [0., 0., self.Jzz]])
        self.xyz = np.array([[0.],
                            [0.],
                            [0.]])
        self.zeta = np.array([[0.],
                            [0.],
                            [0.]])
        self.uvw = np.array([[0.],
                            [0.],
                            [0.]])
        self.pqr = np.array([[0.],
                            [0.],
                            [0.]])
        self.g_q = np.array([[0.],
                            [0.],
                            [0.],
                            [-self.g]])
        self.uvw_q = np.array([[0.],
                                [0.],
                                [0.],
                                [0.]])
        self.pqr_q = np.array([[0.],
                                [0.],
                                [0.],
                                [0.]])
        self.rpm = np.array([0.0, 0., 0., 0.])
        self.uvw_dot = np.array([[0.],
                                [0.],
                                [0.]])
        self.pqr_dot = np.array([[0.],
                                [0.],
                                [0.]])
        
        self.hov_rpm = sqrt((self.mass*self.g)/self.n_motors/self.kt)
        self.max_rpm = sqrt(1./self.hov_p)*self.hov_rpm
        self.q = self.euler_to_q(self.zeta)
        
    def set_state(self, xyz, zeta, uvw, pqr):
        """
            Sets the state space of our vehicle
        """

        self.xyz = xyz
        self.zeta = zeta
        self.q = self.euler_to_q(zeta)
        self. uvw = uvw
        self.pqr = pqr
    
    def get_state(self):
        """
            Returns the current state space
        """

        return self.xyz, self.zeta, self.q, self.uvw, self.pqr
    
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

    def q_norm(self, v):
        """
            Quaternion rotations rely on a unit quaternion. To ensure
            this is the case, we normalize here.
        """

        return v/np.linalg.norm(v)
    
    def q_mult(self, p):
        """
            One way to compute the Hamilton product is usin Q(p)q, where Q(p) is
            the below 4x4 matrix, and q is a 4x1 quaternion. I decided not to do
            the full multiplication here, and instead return Q(p).  
        """

        p0, p1, p2, p3 = p[0,0], p[1,0], p[2,0], p[3,0]
        return np.array([[p0, -p1, -p2, -p3],
                        [p1, p0, -p3, p2],
                        [p2, p3, p0, -p1],
                        [p3, -p2, p1, p0]])

    def q_conj(self, q):
        """
            Returns the conjugate of quaternion q. q* = q'/|q|, where q is the
            magnitude, and q' is the inverse [p0, -p1, -p2, -p3]^T. Since we
            always normalize after updating q, we should always have a unit
            quaternion. This means we don't have to normalize in this routine.
        """

        p0, p1, p2, p3 = q
        return np.array([p0, 
                        -p1, 
                        -p2, 
                        -p3])
    
    def q_to_euler(self, q):
        """
            Convert quaternion q to a set of angles zeta. We do all of the heavy
            lifting with quaternions, and then return the Euler angles since they
            are more intuitive.
        """

        q0, q1, q2, q3 = q
        phi = atan2(2.*(q0*q1+q2*q3),q0**2-q1**2-q2**2+q3**2)
        theta = asin(2.*q0*q2-q3*q1)
        psi = atan2(2.*(q0*q3+q1*q2),q0**2+q1**2-q2**2-q3**2)
        return np.array([phi,
                        theta,
                        psi])
    
    def euler_to_q(self, zeta):
        """
            Converts a set of Euler angles to a quaternion. We do this at the very
            start, since we initialize the vehicle with Euler angles zeta.
        """
        
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
            return -(self.kd*mag**2)*norm

    def aero_moments(self):
        """
            Models aero moments in the body xyz axis as a function of angular velocity
        """

        mag = np.linalg.norm(self.pqr)
        if mag == 0:
            return np.array([[0.0],[0.0],[0.0]])
        else:
            norm = self.pqr/mag
            return -(self.km*mag**2)*norm

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
            Updating the EOMs using second order leapfrog integration (kick-drift-kick
            form) with quaternion rotations. Should be far more accurate than quadrotor,
            and the quaternion rotations should be both faster and avoid the singularity
            at pitch +-90 degrees.
        """
        
        rpm = np.clip(rpm, 0., self.max_rpm)
        
        # thrust forces and moments, aerodynamic forces and moments
        ft = self.thrust_forces(rpm)
        mt = self.thrust_moments(rpm)
        fa = self.aero_forces()
        ma = self.aero_moments()

        # calc angular momentum
        H = self.J.dot(self.pqr)
        
        # rotate gravity vector from inertial frame to body frame using qpq^-1
        Q = self.q_mult(self.q)
        Q_inv = self.q_conj(self.q)
        g_b = Q.dot(self.q_mult(self.g_q).dot(Q_inv))[1:]

        # linear and angular accelerations due to thrust and aerodynamic effects
        uvw_dot = (ft+fa)/self.mass+g_b-np.cross(self.pqr, self.uvw, axis=0)
        pqr_dot = np.linalg.inv(self.J).dot(((mt+ma)-np.cross(self.pqr, H, axis=0)))
        
        # kick: v_{i+0.5} = v_{i}+a_{i}dt/2
        self.uvw += self.uvw_dot*self.dt/2.
        self.pqr += self.pqr_dot*self.dt/2.
        self.uvw_q[1:] = self.uvw[:]
        self.pqr_q[1:] = self.pqr[:]

        # drift: x_{i+1} = x_{i}+v_{i+0.5}dt
        q_dot = -0.5*Q.dot(self.pqr_q)
        self.q = self.q_norm(self.q+q_dot*self.dt)
        xyz_dot = self.q_mult(Q_inv).dot(self.q_mult(self.uvw_q).dot(self.q))[1:]
        self.xyz += xyz_dot*self.dt
        self.zeta = self.q_to_euler(self.q)

        # kick: v_{i+1} = v_{i+0.5}+a_{i+1}dt/2
        self.uvw += uvw_dot*self.dt/2.
        self.pqr += pqr_dot*self.dt/2.
        self.uvw_dot = uvw_dot
        self.pqr_dot = pqr_dot
        if not return_acceleration:
            return self.xyz, self.zeta, self.q, self.uvw, self.pqr
        else:    
            return self.xyz, self.zeta, self.q, self.uvw, self.pqr, xyz_dot, q_dot, uvw_dot, pqr_dot