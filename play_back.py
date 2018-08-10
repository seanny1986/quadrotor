import torch
import os
import matplotlib.pyplot as plt
import gym
import gym_aero
import numpy as np

##WORKS with ddpg, ppo gae and trpo only!!

directory = os.getcwd()
fp = directory + "/saved_policies/ddpg-Hover-v0.pth.tar"
dyn = torch.load(fp,map_location='cpu')
alg = 'ddpg'

def main():
    env = gym.make('Hover-v0')

    steps = 50
    __Tensor = torch.Tensor
    state = np.array(env.reset(),dtype="float32")
    add = np.array([0. ,0. ,0., 0])
    s_torch = torch.from_numpy(state.copy())

    if alg == 'gae' or alg == 'ppo':
        action, log_prob, value  = dyn.select_action(s_torch)
    else:  #for ddpg and trpo only
        action = dyn.select_action(s_torch)

    env.render()


    for i in range(0, steps):
        state, reward, done, _  = env.step(action.cpu().numpy())
        state = __Tensor(state)
        if alg == 'gae' or alg == 'ppo':
            action, log_prob, value  = dyn.select_action(s_torch)
        else:  #for ddpg and trpo only
            action = dyn.select_action(s_torch)
        env.render()

        if done:
            break

if __name__ == "__main__":
    main()
