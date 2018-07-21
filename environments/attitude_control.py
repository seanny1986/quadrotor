import simulation.quadrotor3 as quad
import simulation.config as cfg
import simulation.animation as ani
import matplotlib.pyplot as pl
import numpy as np
import random
from math import pi, sin, cos


"""
    Environment wrapper for a hover task. The goal of this task is for the agent to climb from [0, 0, 0]^T
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
        self.goal_uvw = np.array([[0.],
                                [0.],
                                [0.]])
        self.goal_pqr = np.array([[0.],
                                [0.],
                                [0.]])

        self.t = 0
        self.T = 5
        self.action_space = 4
        self.observation_space = 12+self.action_space+12

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

        self.iris.set_state(self.goal_xyz, self.goal_zeta, self.goal_uvw, self.goal_pqr)
        xyz, zeta, uvw, pqr = self.iris.get_state()

        self.vec_xyz = xyz-self.goal_xyz
        self.vec_zeta = zeta-self.goal_zeta
        self.vec_uvw = uvw-self.goal_uvw
        self.vec_pqr = pqr-self.goal_pqr

        self.dist_sq_xyz = np.linalg.norm(self.vec_xyz)**2
        self.dist_sq_zeta = np.linalg.norm(self.vec_zeta)**2
        self.dist_sq_uvw = np.linalg.norm(self.vec_uvw)**2
        self.dist_sq_pqr = np.linalg.norm(self.vec_pqr)**2

    def init_rendering(self):

        # rendering parameters
        pl.close("all")
        pl.ion()
        self.fig = pl.figure("Hover")
        self.axis3d = self.fig.add_subplot(111, projection='3d')
        self.vis = ani.Visualization(self.iris, 6, quaternion=True)

    def reward(self, state, action):
        xyz, zeta, uvw, pqr = state
        
        self.vec_xyz = xyz-self.goal_xyz
        self.vec_zeta = zeta-self.goal_zeta
        self.vec_uvw = uvw-self.goal_uvw
        self.vec_pqr = pqr-self.goal_pqr

        self.dist_sq_xyz = np.linalg.norm(self.vec_xyz)**2
        self.dist_sq_zeta = np.linalg.norm(self.vec_zeta)**2
        self.dist_sq_uvw = np.linalg.norm(self.vec_uvw)**2
        self.dist_sq_pqr = np.linalg.norm(self.vec_pqr)**2

        dist_rew = -self.dist_sq_xyz
        att_rew = -self.dist_sq_zeta
        vel_rew = -self.dist_sq_uvw
        ang_rew = -self.dist_sq_pqr
        ctrl_rew = -np.sum(((action/self.action_bound[1])**2))
        time_rew = 1.
        return dist_rew, att_rew, vel_rew, ang_rew, ctrl_rew, time_rew

    def terminal(self, pos):
        xyz, zeta = pos
        mask1 = zeta > pi/2
        mask2 = zeta < -pi/2
        mask3 = np.abs(xyz) > 3
        if np.sum(mask1) > 0 or np.sum(mask2) > 0 or np.sum(mask3) > 0:
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
        next_state = sinx+cosx+uvw.T.tolist()[0]+pqr.T.tolist()[0]+(action/self.action_bound[1]).tolist()
        info = self.reward((xyz, zeta, uvw, pqr), action)
        done = self.terminal((xyz, zeta))
        reward = sum(info)
        goals = self.vec_xyz.T.tolist()[0]+self.vec_zeta.T.tolist()[0]+self.vec_uvw.T.tolist()[0]+self.vec_pqr.T.tolist()[0]
        next_state = [next_state+goals]
        self.t += self.ctrl_dt
        return next_state, reward, done, info

    def reset(self):
        self.t = 0.
        self.iris.set_state(self.goal_xyz, self.goal_zeta, self.goal_uvw, self.goal_pqr)
        xyz, zeta, uvw, pqr = self.iris.get_state()
        self.vec_xyz = xyz-self.goal_xyz
        self.vec_zeta = zeta-self.goal_zeta
        self.vec_uvw = uvw-self.goal_uvw
        self.vec_pqr = pqr-self.goal_pqr
        tmp = zeta.T.tolist()[0]
        action = [x/self.action_bound[1] for x in self.trim]
        sinx = [sin(x) for x in tmp]
        cosx = [cos(x) for x in tmp]
        goals = self.vec_xyz.T.tolist()[0]+self.vec_zeta.T.tolist()[0]+self.vec_uvw.T.tolist()[0]+self.vec_pqr.T.tolist()[0]
        state = [sinx+cosx+uvw.T.tolist()[0]+pqr.T.tolist()[0]+action+goals]
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


