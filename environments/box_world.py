import simulation.quadrotor as quad
import simulation.config as cfg
import envs
import random

class Environment:
    def __init__(self, size=25, n_boxes=15, b_max=1.5):
        self.size = size
        self.n_boxes = n_boxes
        self.b_max = b_max
        self.obstacles, self.lower_bnd, self.upper_bnd = self.generate_map()

        self.goal_thresh = 0.1
        self.goal = None
        self.params = cfg.params
        self.iris = quad.Quadrotor(self.params)
    
    def generate_map(self):
        # TODO: ensure boxes don't immediately impinge on the aircraft
        obs = [[random.uniform(0., self.size) for x in range(3)] for b in range(self.n_boxes)]
        sizes = [random.uniform(0., self.b_max) for s in range(self.n_boxes)]
        lower_bnd = [[x-s/2. for x in y] for y, s in zip(obs,sizes)]
        upper_bnd = [[x+s/2. for x in y] for y, s in zip(obs,sizes)]
        return obs, lower_bnd, upper_bnd
    
    def detect_collision(self, pos):
        col = [[x>l and x<u for x, y, z in zip(pos, l, u)] for l, u in zip(self.lower_bnd, self.upper_bnd)]
        return sum(col)>0

    def reward(self, pos):
        # TODO: implement reward
        return None

    def terminal(self, pos):
        col = self.detect_collision(pos)
        dist_sq = [(x-g)**2 for x, g in zip(pos, self.goal)]
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

