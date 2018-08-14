import torch
import os
import matplotlib.pyplot as plt
import gym
import gym_aero
import numpy as np
import algs.ind.gae as gae
import utils

##WORKS with ddpg, ppo gae and trpo only!!

directory = os.getcwd()
fp = directory + "/saved_policies/gae-Hover-v0.pth.tar"
alg = 'gae'

def main():
    env = gym.make('Hover-v0')
    hidden_dim = 64
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    #pi = ppo.ActorCritic(state_dim, hidden_dim, action_dim)
    #beta = ppo.ActorCritic(state_dim, hidden_dim, action_dim)
    network_settings = {
                                "gamma": 0.99,
                                "lambda": 0.92
                                }

    agent = gae.GAE(state_dim, hidden_dim, action_dim, network_settings, GPU=False)
    utils.load(agent, fp)
    __Tensor = torch.Tensor
    state = np.array(env.reset(), dtype="float32")
    s_torch = torch.from_numpy(state.copy())
    if alg == 'gae' or alg == 'ppo':
        action, _, _  = agent.select_action(s_torch)
    else:  #for ddpg and trpo only
        action = agent.select_action(s_torch)
    env.render()
    done = False
    running_reward = 0
    while not done:
        state, reward, done, _  = env.step(action.cpu().numpy())
        running_reward += reward
        state = __Tensor(state)
        if alg == 'gae' or alg == 'ppo':
            action, _, _  = agent.select_action(state)
        else:  #for ddpg and trpo only
            action = agent.select_action(state)
        env.render()

        if done:
            break
    print("Running reward: {:.5f}".format(running_reward))

if __name__ == "__main__":
    main()
