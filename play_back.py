import torch
import os
import matplotlib.pyplot as plt
import gym
import gym_aero
import numpy as np
import algs.ind.ppo as ppo
import utils
import argparse
import config as cfg

"""
    function to play back saved policies.

    -- Sean Morrison, 2018
"""

parser = argparse.ArgumentParser(description="PyTorch actor-critic example")
parser.add_argument("--env", type=str, default="Hover", metavar="E", help="environment to run")
parser.add_argument("--pol", type=str, default="ppo", metavar="P", help="policy to run")
args = parser.parse_args()

directory = os.getcwd()
fp = directory + "/saved_policies/"+args.pol+"-"+args.env+"-v0.pth.tar"

def main():
    env_name = args.env+"-v0"
    env = gym.make(env_name)
    agent = utils.load(fp)
    state = torch.Tensor(env.reset())
    env.render()
    done = False
    running_reward = 0
    while not done:
        action  = agent.select_action(state)
        if isinstance(action, tuple):
            action = action[0]
        state, reward, done, _  = env.step(action.cpu().numpy())
        running_reward += reward
        state = torch.Tensor(state)
        env.render()
        if done:
            break
    print("Running reward: {:.3f}".format(running_reward))

if __name__ == "__main__":
    main()