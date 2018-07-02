import simulation.quadrotor as quad
import simulation.config as cfg
import simulation.animation as ani
import numpy as np
import random
from math import pi, sin, cos

class Environment:
    """
        Environment where we spawn boxes randomly and a LIDAR-equipped aircraft must navigate them
        to get to a given goal. Still under construction.
    """
    def __init__(self, size=25, n_boxes=15, b_max=1.5):
        self.size = size
        self.b_max = b_max
        self.goal_thresh = 0.1
        self.goal = None        
        self.params = cfg.params
        self.iris = quad.Quadrotor(self.params)
        self.obs_xyz, self.lower_bnd, self.upper_bnd = self.generate_map()
        
        self.lower_upper = zip(self.lower_bnd, self.upper_bnd)
        self.box_range = range(n_boxes)

        self.params = cfg.params
        self.iris = quad.Quadrotor(self.params)

        self.lidar_rad = 1.5
        self.lidar_sweep = pi
        self.lidar_lines = 15
        self.lidar_spacing = [self.lidar_lines*pi*(i/self.lidar_lines) for i in range(self.lidar_lines)]
        self.lidar_ds = [np.array([self.lidar_rad*cos(th),self.lidar_rad*sin(th),0.]) for th in self.lidar_spacing]
    
        self.observation_space = 12+self.lidar_lines+3
        self.action_space = 4

        self.vec = None
        self.dist_sq = None

    def generate_map(self):
        # TODO: ensure boxes don't immediately impinge on the aircraft, ensure goal is reachable
        obs_xyz = [[random.uniform(-self.size/2., self.size/2.) for x in range(3)] for b in self.box_range]
        obs_sizes = [random.uniform(0., self.b_max) for s in self.box_range]
        xyz_sizes = zip(obs_xyz, obs_sizes)
        lower_bnd = [[x-s/2. for x in y] for y, s in xyz_sizes]
        upper_bnd = [[x+s/2. for x in y] for y, s in xyz_sizes]
        return obs_xyz, lower_bnd, upper_bnd
    
    def detect_collision(self, pos):
        col = [[x>y and x<z for x, y, z in zip(pos, l, u)] for l, u in self.lower_upper]
        return sum(col) > 0
    
    def detect_intersection(self, pos, ds):
        # TODO: finish logic here
        l1 = pos
        l2 = pos+ds
        for l, u in self.lower_upper:
            if (l2[0] < l[0] and l1[0] < l[0]):
                return False
            if (l2[0] > u[0] and l1[0] > u[0]):
                return False
            if (l2[1] < l[1] and l1[1] < l[1]):
                return False
            if (l2[1] > u[1] and l1[1] > u[1]):
                return False
            if (l2[2] < l[2] and l1[2] < l[2]):
                return False
            if (l2[2] > u[2] and l1[2] > u[2]):
                return False
            if self.detect_collision(l2):
                return True

        sect = [[x>y for x, y in zip(pos, l)] for l in self.lower_bnd]
    
    def lidar(self, pos, zeta):
        # TODO: finish logic here
        R = self.iris.R1(zeta).T
        rotated = [R.dot(x.T) for x in self.lidar_ds]
        
        return None

    def reward(self, pos):
        # TODO: implement reward
        return None

    def terminal(self, pos):
        col = self.detect_collision(pos.T.tolist())
        dist_sq = [(x-g)**2 for x, g in zip(pos, self.goal.T.tolist())]
        goal_achieved = sum([self.goal_thresh > x for x in dist_sq])
        if goal_achieved > 0.: goal_achieved = True
        if col or goal_achieved:
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
        # TODO: implement goal generator
        pass

