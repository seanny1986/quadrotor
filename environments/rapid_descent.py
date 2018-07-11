import simulation.quadrotor2 as quad
import simulation.config as cfg
import simulation.animation as ani
import matplotlib.pyplot as pl
import numpy as np
import random
from math import pi, sin, cos

class Environment:
    """
        Implements a rapid descent environment for learning control policies. At this stage, the environment doesn't
        include a high-fidelity rotor model. Ideally, the simulation used here should use a rotor model that includes
        the effects of vortex ring state, and propwash through the disk. I'm guessing a corrected BEMT model could be
        used for this.
        At the moment, the model assumes propeller thrust is a function ft = kw^2. We check for vortex ring state by
        limiting the aircraft to less than -2m/s body-z axis.
    """
    
    def __init__(self):
        
        # environment parameters
        self.goal_xyz = np.array([[0.],
                                [0.],
                                [0.]])
        self.goal_xyz_dot = np.array([[0.],
                                    [0.],
                                    [0.]])
        self.start_xyz = np.array([[0.],
                                    [0.],
                                    [50.]])
        self.start_zeta = np.array([[0.],
                                    [0.],
                                    [0.]])
        self.start_uvw = np.array([[0.],
                                    [0.],
                                    [0.]])
        self.start_pqr = np.array([[0.],
                                    [0.],
                                    [0.]])
        self.goal_thresh_xyz = 0.05
        self.t = 0
        self.T = 15
        self.r = 1.5
        self.action_space = 4
        self.observation_space = 18+self.action_space

        # simulation parameters
        self.params = cfg.params
        self.iris = quad.Quadrotor(self.params)
        self.sim_dt = self.params["dt"]
        self.ctrl_dt = 0.05
        self.steps = range(int(self.ctrl_dt/self.sim_dt))
        self.action_bound = [0, self.iris.max_rpm]
        self.H = int(self.T/self.ctrl_dt)
        self.hov_rpm = self.iris.hov_rpm
        self.trim = [self.hov_rpm, self.hov_rpm, self.hov_rpm, self.hov_rpm]

        self.vec_xyz = None
        self.vec_uvw = None
        self.dist_sq = None
        self.goal_achieved = False
    
    def init_rendering(self):
        # rendering parameters
        pl.close("all")
        pl.ion()
        self.fig = pl.figure("Rapid Descent")
        self.axis3d = self.fig.add_subplot(111, projection='3d')
        self.vis = ani.Visualization(self.iris, 6)

    def reward(self, xyz, xyz_dot, action):
        self.vec_xyz = xyz-self.goal_xyz
        self.vec_xyz_dot = xyz_dot-self.goal_xyz
        self.dist_sq = np.linalg.norm(self.vec_xyz)
        self.vel_sq = np.linalg.norm(self.vec_xyz_dot)
        dist_rew = np.exp(-self.dist_sq)
        vel_rew = np.exp(-self.vel_sq)
        ctrl_rew = -np.sum((action**2))/400000.
        cmplt_rew = 0.
        if self.dist_sq < self.goal_thresh_xyz:
            cmplt_rew = 1000.
            self.goal_achieved = True
        return dist_rew+vel_rew+ctrl_rew+cmplt_rew

    def terminal(self, pos):
        xyz, zeta, uvw = pos
        mask1 = zeta[0:2] > pi/2
        mask2 = zeta[0:2] < -pi/2
        mask3 = np.abs(xyz[0:2]) > 10
        mask4 = xyz[2] < 0
        mask5 = uvw[2] < -2.
        if np.sum(mask1) > 0 or np.sum(mask2) > 0 or np.sum(mask3) > 0 or mask4 > 0 or mask5 > 0:
            return True
        elif self.goal_achieved:
            print("Goal Achieved!")
            return True
        elif self.t == self.T:
            print("Sim time reached")
        else:
            return False

    def step(self, action):
        for _ in self.steps:
            xyz, zeta, uvw, pqr, xyz_dot, _, _, _ = self.iris.step(action, return_acceleration=True)
        tmp = zeta.T.tolist()[0]
        sinx = [sin(x) for x in tmp]
        cosx = [cos(x) for x in tmp]
        next_state = sinx+cosx+uvw.T.tolist()[0]+pqr.T.tolist()[0]+action.tolist()[0]
        reward = self.reward(xyz, xyz_dot, action)
        done = self.terminal((xyz, zeta, uvw))
        info = None
        next_state = [next_state+self.vec_xyz.T.tolist()[0]+self.vec_xyz_dot.T.tolist()[0]]
        self.t += self.ctrl_dt
        return next_state, reward, done, info

    def reset(self):
        self.goal_achieved = False
        self.t = 0.
        self.iris.set_state(self.start_xyz, self.start_zeta, self.start_uvw, self.start_pqr)
        xyz, zeta, _, uvw, pqr = self.iris.get_state()
        self.vec_xyz = xyz-self.goal_xyz
        tmp = zeta.T.tolist()[0]
        sinx = [sin(x) for x in tmp]
        cosx = [cos(x) for x in tmp]
        next_state = sinx+cosx+uvw.T.tolist()[0]+pqr.T.tolist()[0]+self.trim
        state = [next_state+self.vec_xyz.T.tolist()[0]+self.vec_xyz_dot.T.tolist()[0]]
        return state
    
    def render(self):
        pl.figure(0)
        self.axis3d.cla()
        self.vis.draw3d(self.axis3d)
        self.vis.draw_goal(self.axis3d, self.goal_xyz)
        self.axis3d.set_xlim(-3, 3)
        self.axis3d.set_ylim(-3, 3)
        self.axis3d.set_zlim(0, 6)
        self.axis3d.set_xlabel('West/East [m]')
        self.axis3d.set_ylabel('South/North [m]')
        self.axis3d.set_zlabel('Down/Up [m]')
        self.axis3d.set_title("Time %.3f s" %(self.t))
        pl.pause(0.001)
        pl.draw()


