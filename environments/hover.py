import simulation.quadrotor as quad
import simulation.config as cfg

class Environment:
    def __init__(self, args):
        self.params = cfg.params
        self.iris = quad.Quadrotor(self.params)
        self.goal = [3., 0., 0.]
        self.goal_thresh = 0.1

    def reward(self, pos):
        # TODO: implement reward
        return None

    def terminal(self,pos):
        dist_sq = [(x-g)**2 for x, g in zip(pos, self.goal)]
        goal_achieved = sum([self.goal_thresh > x for x in dist_sq])
        if goal_achieved > 0.: goal_achieved = True
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