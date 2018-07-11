import simulation.quadrotor as quad
import simulation.config as cfg
import simulation.animation as ani
import matplotlib.pyplot as pl
import numpy as np
import random
from math import pi, sin, cos

class Environment:
    """
        Environment wrapper for learning perching. The goal is for the aircraft to reach the goal perch
        state on the wall with zero velocity, with the normal vector of the aircraft facing in the same 
        direction as the normal vector of the wall. 
    """

    def __init__(self):
        
        # environment parameters
        self.goal_xyz = np.array([[0.],
                                [0.],
                                [1.5]])
        self.goal_zeta = np.array([[0.],
                                [pi/2],
                                [0.]])
        self.goal_xyz_dot = np.array([[0.],
                                    [0.],
                                    [0.]])
        self.goal_thresh = 0.05
        self.t = 0
        self.T = 15
        self.r = 1.5
        self.action_space = 4
        self.observation_space = 15

        # simulation parameters
        self.params = cfg.params
        self.iris = quad.Quadrotor(self.params)
        self.sim_dt = self.params["dt"]
        self.ctrl_dt = 0.05
        self.steps = range(int(self.ctrl_dt/self.sim_dt))
        self.action_bound = [0, self.iris.max_rpm]
        self.H = int(self.T/self.ctrl_dt)

        self.vec = None
        self.dist_sq = None
        self.goal_achieved = False
    
    def init_rendering(self):
        # rendering parameters
        pl.close("all")
        pl.ion()
        self.fig = pl.figure("perching")
        self.axis3d = self.fig.add_subplot(111, projection='3d')
        self.vis = ani.Visualization(self.iris, 6)

    def reward(self, xyz, action):
        self.vec = self.iris.xyz-self.goal_xyz
        self.dist_sq = np.linalg.norm(self.vec)
        dist_rew = np.exp(-self.dist_sq)
        ctrl_rew = -np.sum((action**2))/400000.
        cmplt_rew = 0.
        if self.dist_sq < self.goal_thresh:
            cmplt_rew = 1000.
            self.goal_achieved = True
        return dist_rew+ctrl_rew+cmplt_rew

    def terminal(self, pos):
        xyz, zeta = pos
        mask1 = zeta[0:2] > pi/2
        mask2 = zeta[0:2] < -pi/2
        mask3 = np.abs(xyz) > 6
        if np.sum(mask1) > 0 or np.sum(mask2) > 0 or np.sum(mask3) > 0:
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
            xyz, zeta, uvw, pqr = self.iris.step(np.array(action))
        tmp = zeta.T.tolist()[0]
        next_state = [sin(x) for x in tmp]+[cos(x) for x in tmp]+uvw.T.tolist()[0]+pqr.T.tolist()[0]
        reward = self.reward(xyz, action)
        done = self.terminal((xyz, zeta))
        info = None
        self.t += self.ctrl_dt
        next_state = [next_state+self.vec.T.tolist()[0]]
        return next_state, reward, done, info

    def reset(self):
        self.goal_achieved = False
        self.t = 0.
        xyz, zeta, uvw, pqr = self.iris.reset()
        self.vec = xyz-self.goal
        tmp = zeta.T.tolist()[0]
        state = [[sin(x) for x in tmp]+[cos(x) for x in tmp]+uvw.T.tolist()[0]+pqr.T.tolist()[0]+self.vec.T.tolist()[0]]
        return state
    
    def render(self):
        pl.figure(0)
        self.axis3d.cla()
        self.vis.draw3d(self.axis3d)
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


