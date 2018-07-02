import simulation.quadrotor2 as quad
import simulation.config as cfg
import simulation.animation as ani
import matplotlib.pyplot as pl
import numpy as np
import random
from math import pi, sin, cos

class Environment:
    def __init__(self):
        
        # environment parameters
        self.goal = np.array([[0.],
                            [0.],
                            [1.5]])
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

        # rendering parameters
        pl.close("all")
        pl.ion()
        self.fig = pl.figure(0)
        self.axis3d = self.fig.add_subplot(111, projection='3d')
        self.vis = ani.Visualization(self.iris, 6)

    def step(self, action):
        for _ in self.steps:
            xyz, zeta, q, uvw, pqr = self.iris.step(np.array(action))
        next_state = [xyz.T.tolist()[0]+zeta.T.tolist()[0]+uvw.T.tolist()[0]+pqr.T.tolist()[0]]
        self.t += self.ctrl_dt
        return next_state, None, None, None

    def reset(self):
        self.goal_achieved = False
        self.t = 0.
        xyz, zeta, q, uvw, pqr = self.iris.reset()
        state = [xyz.T.tolist()[0]+zeta.T.tolist()[0]+uvw.T.tolist()[0]+pqr.T.tolist()[0]]
        return state
    
    def set_state(self, xyz, zeta, uvw, pqr):
        self.iris.set_state(xyz, zeta, uvw, pqr)
    
    def get_aircraft_state(self):
        return self.iris.get_state()
    
    def render(self):
        pl.figure(0)
        self.axis3d.cla()
        self.vis.draw3d(self.axis3d)
        self.axis3d.set_xlim(-3, 3)
        self.axis3d.set_ylim(-3, 3)
        self.axis3d.set_zlim(0, 6)
        self.axis3d.set_xlabel('West/East [m]')
        self.axis3d.set_ylabel('South/North [m]')
        self.axis3d.set_zlabel('Down/Up [m]')
        self.axis3d.set_title("Time %.3f s" %(self.t))
        pl.pause(0.001)
        pl.draw()


