import simulation.quadrotor2 as quad
import simulation.config as cfg
import simulation.animation as ani
import matplotlib.pyplot as pl
import numpy as np
import random
from math import pi, sin, cos

class Environment:
    """
        Environment wrapper for training aircraft transition models. This might not seem particularly
        useful (you can easily do this without the wrapper), but it cleans things up a fair bit. Can
        set deterministic or non-deterministic starting conditions. Deterministic conditions make the
        model easy to learn, but makes the model less useful overall.
    """
    
    def __init__(self):
        
        # environment parameters
        self.deterministic_s0 = True
        self.t = 0
        self.T = 15
        self.r = 1.5
        self.action_space = 4
        self.observation_space = 15+self.action_space

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

        # define bounds here
        self.xzy_bound = 1.
        self.zeta_bound = pi/2
        self.uvw_bound = 10
        self.pqr_bound = 1.

        # rendering parameters
        pl.close("all")
        pl.ion()
        self.fig = pl.figure(0)
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

    def step(self, action):
        for _ in self.steps:
            xyz, zeta, q, uvw, pqr = self.iris.step(np.array(action))
        tmp = zeta.T.tolist()[0]
        sinx = [sin(x) for x in tmp]
        cosx = [cos(x) for x in tmp]
        next_state = [sinx+cosx+uvw.T.tolist()[0]+pqr.T.tolist()[0]+action.tolist()[0]]
        self.t += self.ctrl_dt
        return next_state, None, None, None

    def reset(self):
        self.t = 0.
        if self.deterministic_s0:
            xyz, zeta, _, uvw, pqr = self.iris.reset()
        else:
            xyz, zeta, uvw, pqr = self.generate_s0()
            self.iris.set_state(xyz, zeta, uvw, pqr)
        action = self.trim
        tmp = zeta.T.tolist()[0]
        sinx = [sin(x) for x in tmp]
        cosx = [cos(x) for x in tmp]
        state = [sinx+cosx+uvw.T.tolist()[0]+pqr.T.tolist()[0]+action]
        return state

    def set_random(self):
        pass

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


