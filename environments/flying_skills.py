import simulation.quadrotor3 as quad
import simulation.config as cfg
import simulation.animation as ani
import matplotlib.pyplot as pl
import numpy as np
import random
from math import pi, sin, cos

class Environment:
    """
        Environment wrapper for training low-level flying skills. In this environment, the aircraft
        has a deterministic starting state by default. We can switch it to have non-deterministic 
        initial states. This is obviously much harder.
    """
    def __init__(self):
        
        # environment parameters
        self.deterministic_s0 = True
        self.goal = self.generate_goal(1.5)
        self.goal_thresh = 0.1
        self.t = 0
        self.T = 2.5
        self.r = 1.5
        self.action_space = 4
        self.observation_space = 15+self.action_space

        # simulation parameters
        self.params = cfg.params
        self.iris = quad.Quadrotor(self.params)
        self.ctrl_dt = 0.05
        self.sim_dt = self.params["dt"]
        self.steps = range(int(self.ctrl_dt/self.sim_dt))
        self.hov_rpm = self.iris.hov_rpm
        self.trim = [self.hov_rpm, self.hov_rpm,self.hov_rpm, self.hov_rpm]
        self.action_bound = [0, self.iris.max_rpm]
        self.H = int(self.T/self.ctrl_dt)

        # define bounds here
        self.xzy_bound = 1.
        self.zeta_bound = pi/2
        self.uvw_bound = 10
        self.pqr_bound = 1.

        self.vec = None
        self.dist_sq = None
    
    def init_rendering(self):
        # rendering parameters
        pl.close("all")
        pl.ion()
        self.fig = pl.figure("Flying Skills")
        self.axis3d = self.fig.add_subplot(111, projection='3d')
        self.vis = ani.Visualization(self.iris, 6, quaternion=True)

    def set_nondeterministic_s0(self):
        self.deterministic_s0 = False

    def generate_s0(self):
        xyz = np.random.uniform(low=-self.xzy_bound, high=self.xzy_bound, size=(3,1))
        zeta = np.random.uniform(low=-self.zeta_bound, high=self.zeta_bound, size=(3,1))
        uvw = np.random.uniform(low=-self.uvw_bound, high=self.uvw_bound, size=(3,1))
        pqr = np.random.uniform(low=-self.pqr_bound, high=self.pqr_bound, size=(3,1))
        xyz[2,:] = abs(xyz[2,:])
        return xyz, zeta, uvw, pqr

    def reward(self, xyz, action):
        self.vec = xyz-self.goal
        self.dist_sq = np.linalg.norm(self.vec)
        dist_rew = np.exp(-self.dist_sq)
        ctrl_rew = -np.sum((action**2))/1e12
        cmplt_rew = 0.
        if self.dist_sq < self.goal_thresh:
            cmplt_rew = 5.
            self.goal_achieved = True
        time_rew = 0.1
        return dist_rew, ctrl_rew, cmplt_rew, time_rew

    def terminal(self, pos):
        xyz, zeta = pos
        mask1 = zeta > pi/2
        mask2 = zeta < -pi/2
        mask3 = np.abs(xyz) > 6
        if np.sum(mask1) > 0 or np.sum(mask2) > 0 or np.sum(mask3) > 0:
            return True
        if self.goal_achieved:
            return True
        else:
            return False

    def step(self, action):
        for _ in self.steps:
            xyz, zeta, uvw, pqr = self.iris.step(action)
        tmp = zeta.T.tolist()[0]
        sinx = [sin(x) for x in tmp]
        cosx = [cos(x) for x in tmp]
        next_state = sinx+cosx+uvw.T.tolist()[0]+pqr.T.tolist()[0]+(action/self.action_bound[1]).tolist()
        info = self.reward(xyz, action)
        done = self.terminal((xyz, zeta))
        reward = sum(info)
        self.t += self.ctrl_dt
        next_state = [next_state+self.vec.T.tolist()[0]]
        return next_state, reward, done, info

    def reset(self):
        self.goal_achieved = False
        self.t = 0.
        if self.deterministic_s0:
            xyz, zeta, uvw, pqr = self.iris.reset()
        else:
            xyz, zeta, uvw, pqr = self.generate_s0()
            self.iris.set_state(xyz, zeta, uvw, pqr)
        self.goal = self.generate_goal(self.r)
        self.vec = xyz-self.goal
        tmp = zeta.T.tolist()[0]
        action = [x/self.action_bound[1] for x in self.trim]
        sinx = [sin(x) for x in tmp]
        cosx = [cos(x) for x in tmp]
        state = [sinx+cosx+uvw.T.tolist()[0]+pqr.T.tolist()[0]+action+self.vec.T.tolist()[0]]
        return state

    def generate_goal(self, r):
        phi = random.uniform(-2*pi, 2*pi)
        theta = random.uniform(-2*pi, 2*pi)
        x = r*sin(theta)*cos(phi)
        y = r*sin(theta)*sin(phi)
        z = r*cos(theta)
        return np.array([[x], 
                        [y], 
                        [z]])
    
    def render(self):
        pl.figure("Flying Skills")
        self.axis3d.cla()
        self.vis.draw3d_quat(self.axis3d)
        self.vis.draw_goal(self.axis3d, self.goal)
        self.axis3d.set_xlim(-3, 3)
        self.axis3d.set_ylim(-3, 3)
        self.axis3d.set_zlim(0, 6)
        self.axis3d.set_xlabel('West/East [m]')
        self.axis3d.set_ylabel('South/North [m]')
        self.axis3d.set_zlabel('Down/Up [m]')
        self.axis3d.set_title("Time %.3f s" %(self.t))
        pl.pause(0.001)
        pl.draw()


