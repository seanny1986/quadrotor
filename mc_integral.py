import math
from math import pi
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import utils
import numpy as np
import environments.envs as envs
import models.one_step as model
import os

def calc_thrust():
    rpm = env.iris.rpm
    q = env.iris.state[3:7]
    Q_inv = env.iris.q_mult(env.iris.q_conj(q))
    fnm = env.iris.rpm_to_u.dot(rpm**2)
    thrust = np.vstack([env.iris.zero, fnm[0:3].reshape(-1,1)])
    thrust_if = Q_inv.dot(env.iris.q_mult(thrust).dot(q))
    if thrust_if[2] >= weight:
        return 1
    else: 
        return 0

samples = 5000
env = envs.make("model_training")
env.set_nondeterministic_s0()
max_rpm = env.action_bound[1]
action_dim = env.action_space
state_dim = env.observation_space
counter = 0
running = True
H = 1
noise = utils.OUNoise(action_dim, mu=10)
mass = env.iris.mass
weight = mass*9.81
    
for s in range(1, samples+1):
    # set random state
    state = env.reset()
    state_actions = []
    next_states = []
        
    # run trajectory
    loss = 0
    for i in range(1, H+1):
        action = np.array([noise.noise()], dtype="float32")*env.action_bound[1]
        next_state, _, _, _ = env.step(action.reshape(action_dim,))
        counter += calc_thrust()
        state = next_state
    print(s)
print("---Percentage: {:.8f}---".format(counter/(samples*H)))
    