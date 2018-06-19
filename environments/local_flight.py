import simulation.quadrotor as quad
import simulation.config as cfg
import simulation.animation as ani
import matplotlib.pyplot as pl
import numpy as np
import random
from math import pi, sin, cos

last_xyz_dist = None
class Environment:
    def __init__(self, args):
        self.params = cfg.params
        self.iris = quad.Quadrotor(self.params)
        self.vis = ani.Axes3D()
        self.goal = None
        self.goal_thresh = 0.1
        self.r = 1.5
        self.T = 15
        self.action_space = 4
        self.observation_space = 15
        pl.close("all")
        pl.ion()
        self.fig = pl.figure(0)
        self.axis3d = fig.add_subplot(111, projection='3d')


    def reward(self, xyz, action):
        global last_xyz_dist
        dist_sq = [(x-g)**2 for x, g in zip(xyz, self.goal)]
        if last_xyz_dist is not None:
            xyz_loss = err_xyz-last_xyz_dist
        else:
            xyz_loss = -err_xyz
        last_xyz_dist = err_xyz
        action_loss = -(action**2).sum()/400000.
        r = 0.
        if err_xyz < self.goal_thresh:
            r = 1000.
        return xyz_loss+action_loss+r

    def terminal(self, pos):
        xyz, zeta = pos
        mask1 = zeta > pi/2
        mask2 = zeta < -pi/2
        mask3 = np.abs(xyz) > 6
        dist_sq = [(x-g)**2 for x, g in zip(xyz, self.goal)]
        goal_achieved = sum([self.goal_thresh > x for x in dist_sq])
        
        if goal_achieved > 0.: goal_achieved = True
        
        if np.sum(mask1) > 0 or np.sum(mask2) > 0 or np.sum(mask3) > 0:
            return True
        if goal_achieved:
            return True
        else:
            return False

    def step(self, action):
        next_state = self.iris.step(action)
        reward = self.reward(next_state)
        done = self.terminal(next_state)
        info = None
        return next_state, reward, done, info
    
    def reset(self):
        self.iris.reset()
        self.goal = self.generate_goal()   

    def generate_goal(self):
        sph = [random.uniform(-2*pi, 2*pi) for i in range(2)]
        x = self.r*sph[1].sin()*sph[0].cos()
        y = self.r*sph[1].sin()*sph[0].sin()
        z = self.r*sph[1].cos()
        self.goal = [x, y, z]
    
    def render(self, axis3d, goal, t):
        pl.figure(0)
        axis3d.cla()
        self.vis.draw3d(axis3d)
        self.vis.draw_goal(axis3d, goal)
        axis3d.set_xlim(-3, 3)
        axis3d.set_ylim(-3, 3)
        axis3d.set_zlim(0, 6)
        axis3d.set_xlabel('West/East [m]')
        axis3d.set_ylabel('South/North [m]')
        axis3d.set_zlabel('Down/Up [m]')
        axis3d.set_title("Time %.3f s" %(t*dt))
        pl.pause(0.001)
        pl.draw()
    
    def numpy_to_pytorch(xyz, zeta, uvw, pqr):
        xyz = torch.from_numpy(xyz.T).float()
        zeta = torch.from_numpy(zeta.T).float()
        uvw = torch.from_numpy(uvw.T).float()
        pqr = torch.from_numpy(pqr.T).float()
        if args.cuda:
            xyz = xyz.cuda()
            zeta = zeta.cuda()
            uvw = uvw.cuda()
            pqr = pqr.cuda()
        return xyz, zeta, uvw, pqr


