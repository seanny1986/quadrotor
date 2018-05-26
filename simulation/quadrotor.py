import numpy as np
from math import sin, cos, tan, pi

class Quadrotor:
    """
        6DOF rigid body, non-linear EOM solver for a plus configuration quadrotor. Aircraft is modeled
        with an East-North-Up axis system for convenience when plotting. This means thrust is positive
        in the body-axis z-direction, and the gravity vector is negative in the intertial axis 
        z-direction. The aircraft comes with a config file that includes the necessary parameters. These
        are:

        mass = the mass of the vehicle in kg
        prop_radius = the radius of the propellers in meters (this is cosmetic only, no momentum theory)
        max_rpm = the maximum rpm value; we clip rpm if is outside the bound 0 <= rpm <= rpm_max
        l = the length between the centre of mass and the centre of the prop disk (i.e. arm length)
        Jxx = the mass moment of inertia about the x-axis (roll)
        Jyy = the mass moment of inertia about the y-axis (pitch)
        Jzz = the mass moment of inertia about the z-axis (yaw)
        kt = motor thrust coefficient
        kq = motor torque coefficient
        kd1 = linear drag coefficient
        kd2 = angular drag coefficient
        dt = solver time step

        -- Sean Morrison, 2018
    """
    
    def __init__(self, params):
        self.mass = params["mass"]
        self.prop_radius = params["prop_radius"]
        self.max_rpm = params["max_rpm"]
        self.l = params["l"]
        self.Jxx = params["Jxx"]
        self.Jyy = params["Jyy"]
        self.Jzz = params["Jzz"]
        self.kt = params["kt"]
        self.kq = params["kq"]
        self.kd1 = params["kd1"]
        self.kd2 = params["kd2"]
        self.dt = params["dt"]
        self.thrust = None

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
        self.g = np.array([[0.],
                            [0.],
                            [-9.81]])
        self.rpm = np.array([0.0, 0., 0., 0.])
        

    def set_state(self, xyz, zeta, uvw, pqr):
        """
            Sets the state space of our vehicle
        """

        self.xyz = xyz
        self.zeta = zeta
        self. uvw = uvw
        self.pqr = pqr
    
    def get_state(self):
        """
            Returns the current state space
        """
        return self.xyz, self.zeta, self.uvw, self.pqr
    
    def reset(self):
        """
            Resets the initial state of the quadrotor
        """

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
        self.rpm = np.array([0., 0., 0., 0.])
        return self.get_state()

    def R1(self, zeta):
        """
            Rotation matrix converting body frame linear values to the inertial frame.
            This matrix is orthonormal, so to go from the intertial frame to the body
            frame, we can take the transpose of this matrix. That is, R1^-1 = R1^T.
            These rotations are for an East-North-Up axis system, since our inertial
            frame (matplotlib) uses this for plotting.
        """
        
        phi = zeta[0,0]
        theta = zeta[1,0]
        psi = zeta[2,0]
        R_z = np.array([[cos(psi), -sin(psi), 0],
                            [sin(psi), cos(psi), 0],
                            [0., 0., 1.]])
        R_y = np.array([[cos(theta), 0., sin(theta)],
                            [0., 1., 0.],
                            [-sin(theta), 0, cos(theta)]])
        R_x =  np.array([[1., 0., 0.],
                            [0., cos(phi), -sin(phi)],
                            [0., sin(phi), cos(phi)]])
        return R_z.dot(R_y.dot(R_x))

    def R2(self, zeta):
        """
            Rotation matrix converting body frame angular velocities to the inertial frame.
            This uses the East-North-Up axis convention.
        """

        theta = zeta[1,0]
        psi = zeta[2,0]

        x11 = cos(psi)/cos(theta)
        x12 = sin(psi)/cos(theta)
        x13 = 0
        x21 = -sin(psi)
        x22 = cos(psi)
        x23 = 0
        x31 = cos(psi)*tan(theta)
        x32 = sin(psi)*tan(theta)
        x33 = 1
        return np.array([[x11, x12, x13],
                        [x21, x22, x23],
                        [x31, x32, x33]])

    def aero_forces(self):
        """
            Calculates drag in the body xyz axis (E-N-U) due to linear velocity
        """

        mag = np.linalg.norm(self.uvw)
        if mag == 0:
            return np.array([[0.0],[0.0],[0.0]])
        else:
            norm = self.uvw/mag
            return -(self.kd1*mag**2)*norm

    def aero_torques(self):
        """
            Calculates drag in the body xyz axis (E-N-U) due to angular velocity
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

        f_body_x, f_body_y = 0, 0
        f_body_z = np.sum(self.thrust)
        return np.array([[f_body_x],
                        [f_body_y],
                        [f_body_z]])
    
    def thrust_torques(self, rpm):
        """
            Calculates torques about the body xyz axis due to motor thrust and torque
        """

        t_body_x = self.l*(self.thrust[1]-self.thrust[3])
        t_body_y = self.l*(self.thrust[2]-self.thrust[0])
        motor_torques = self.kq*rpm**2
        t_body_z = -motor_torques[0]+motor_torques[1]-motor_torques[2]+motor_torques[3]
        return np.array([[t_body_x],
                        [t_body_y],
                        [t_body_z]])
    
    def step(self, rpm, return_acceleration=False):
        """
            Semi-implicit Euler update of the non-linear equations of motion. Use the
            matrix form since it's much nicer to work with. Our two equations are:
            
            v_dot = F_b/m + R1^{-1}G_i - omega x v
            omega_dot = J^{-1}[Q_b - omega x v]

            Where F_b are the external body forces (thrust+drag) in the body frame, m 
            is the mass of the vehicle, R1^{-1} is the inverse of matrix R1 (since R1
            rotates the body frame to the inertial frame, the inverse rotates the inertial
            to the body frame), G_i is the gravity vector in the inertial frame (0,0,-9.81),
            omega is the angular velocity, v is the velocity, J is the inertia matrix, and
            Q_b are the external moments about the body axes system (motor thrust, motor
            torque, and aerodynamic moments).

            In some cases we may want to return the acceleration, though the default is False.
        """
        
        rpm[rpm < 0] = 0
        rpm[rpm > self.max_rpm] = self.max_rpm
        self.thrust = self.kt*rpm**2
        r1 = self.R1(self.zeta)
        r2 = self.R2(self.zeta)
        fm = self.thrust_forces(rpm)
        tm = self.thrust_torques(rpm)
        fa = self.aero_forces()
        ta = self.aero_torques()
        Jw = self.J.dot(self.pqr)
        uvw_dot = (fm+fa)/self.mass+r1.T.dot(self.g)-np.cross(self.pqr,self.uvw,axis=0)
        pqr_dot = np.linalg.inv(self.J).dot(((tm+ta)-np.cross(self.pqr,Jw,axis=0)))
        self.uvw += uvw_dot*self.dt
        self.pqr += pqr_dot*self.dt
        xyz_dot = r1.dot(self.uvw)
        zeta_dot = r2.dot(self.pqr)
        self.xyz += xyz_dot*self.dt
        self.zeta += zeta_dot*self.dt
        if not return_acceleration:
            return self.xyz, self.zeta, self.uvw, self.pqr
        else:    
            return self.xyz, self.zeta, self.uvw, self.pqr, xyz_dot, zeta_dot, uvw_dot, pqr_dot