import simulation.quadrotor3 as quad
import simulation.config as cfg
import simulation.animation as ani
import matplotlib.pyplot as pl
import numpy as np
import random
from math import pi, sin, cos


"""
    Environment wrapper for a climb & hover task. The goal of this task is for the agent to climb from [0, 0, 0]^T
    to [0, 0, 1.5]^T, and to remain at that altitude until the the episode terminates at T=15s.
"""

class Environment:
    def __init__(self):
        
        # environment parameters
        self.goal_xyz = np.array([[0.],
                                [0.],
                                [1.5]])
        self.goal_zeta = np.array([[0.],
                                [0.],
                                [0.]])
        self.goal_thresh = 0.05
        self.t = 0
        self.T = 5
        self.action_space = 4
        self.observation_space = 15+self.action_space+6

        # simulation parameters
        self.params = cfg.params
        self.iris = quad.Quadrotor(self.params)
        self.sim_dt = self.params["dt"]
        self.ctrl_dt = 0.05
        self.steps = range(int(self.ctrl_dt/self.sim_dt))
        self.action_bound = [0, self.iris.max_rpm]
        self.H = int(self.T/self.ctrl_dt)
        self.hov_rpm = self.iris.hov_rpm
        self.trim = [self.hov_rpm, self.hov_rpm,self.hov_rpm, self.hov_rpm]

        self.vec_xyz = self.iris.get_state()[0]-self.goal_xyz
        self.vec_zeta = self.iris.get_state()[0]-self.goal_zeta
        self.dist_norm = np.linalg.norm(self.vec_xyz)
        self.att_norm = np.linalg.norm(self.vec_zeta)
        self.goal_achieved = False

    def init_rendering(self):
        # rendering parameters
        pl.close("all")
        pl.ion()
        self.fig = pl.figure("Hover")
        self.axis3d = self.fig.add_subplot(111, projection='3d')
        self.vis = ani.Visualization(self.iris, 6, quaternion=True)

    def reward(self, state, action):
        xyz, zeta, _, _ = state
        curr_dist = xyz-self.goal_xyz
        curr_att = zeta-self.goal_zeta
        dist_hat = np.linalg.norm(curr_dist)
        att_hat = np.linalg.norm(curr_att)
        dist_rew = -100*(dist_hat-self.dist_norm)
        att_rew = 10*(att_hat-self.att_norm)
        self.dist_norm = dist_hat
        self.att_norm = att_hat
        self.vec_xyz = curr_dist
        self.vec_zeta = curr_att
        ctrl_rew = -np.sum(((action/self.action_bound[1])**2))
        cmplt_rew = 0.
        if self.dist_norm < self.goal_thresh:
            cmplt_rew = 10.
            self.goal_achieved = True
        time_rew = 0.1
        return dist_rew, ctrl_rew, cmplt_rew, time_rew

    def terminal(self, pos):
        xyz, zeta = pos
        mask1 = 0#zeta > pi/2
        mask2 = 0#zeta < -pi/2
        mask3 = np.abs(xyz) > 3
        if np.sum(mask1) > 0 or np.sum(mask2) > 0 or np.sum(mask3) > 0:
            return True
        elif self.goal_achieved:
            #print("Goal Achieved!")
            return True
        elif self.t == self.T:
            print("Sim time reached")
            return True
        else:
            return False

    def step(self, action):
        for _ in self.steps:
            xyz, zeta, uvw, pqr = self.iris.step(action)
        tmp = zeta.T.tolist()[0]
        sinx = [sin(x) for x in tmp]
        cosx = [cos(x) for x in tmp]
        a = (self.iris.rpm/self.action_bound[1]).tolist()
        next_state = xyz.T.tolist()[0]+sinx+cosx+uvw.T.tolist()[0]+pqr.T.tolist()[0]+a
        info = self.reward((xyz, zeta, uvw, pqr), action)
        done = self.terminal((xyz, zeta))
        reward = sum(info)
        goal = self.vec_xyz.T.tolist()[0]+self.vec_zeta.T.tolist()[0]
        next_state = [next_state+goal]
        self.t += self.ctrl_dt
        return next_state, reward, done, info

    def reset(self):
        self.goal_achieved = False
        self.t = 0.
        xyz, zeta, uvw, pqr = self.iris.reset()
        self.vec_xyz = xyz-self.goal_xyz
        self.vec_zeta = zeta-self.goal_zeta
        tmp = zeta.T.tolist()[0]
        sinx = [sin(x) for x in tmp]
        cosx = [cos(x) for x in tmp]
        a = (self.iris.rpm/self.action_bound[1]).tolist()
        goal = self.vec_xyz.T.tolist()[0]+self.vec_zeta.T.tolist()[0]
        state = [xyz.T.tolist()[0]+sinx+cosx+uvw.T.tolist()[0]+pqr.T.tolist()[0]+a+goal]
        return state
    
    def render(self):
        pl.figure("Hover")
        self.axis3d.cla()
        self.vis.draw3d_quat(self.axis3d)
        self.vis.draw_goal(self.axis3d, self.goal_xyz)
        self.axis3d.set_xlim(-3, 3)
        self.axis3d.set_ylim(-3, 3)
        self.axis3d.set_zlim(-3, 3)
        self.axis3d.set_xlabel('West/East [m]')
        self.axis3d.set_ylabel('South/North [m]')
        self.axis3d.set_zlabel('Down/Up [m]')
        self.axis3d.set_title("Time %.3f s" %(self.t))
        pl.pause(0.001)
        pl.draw()


