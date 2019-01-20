import torch
import os
import gym
import gym_aero
import numpy as np
import utils
import argparse
import config as cfg
from math import sin, cos, tan, pi

"""
    Function to play back saved policies and save video. I hate matplotlib.

    -- Sean Morrison, 2018
"""

# script arguments. E.g. python play_back.py --env="Hover" --pol="ppo"
parser = argparse.ArgumentParser(description="PyTorch actor-critic example")
parser.add_argument("--env", type=str, default="Hover", metavar="E", help="environment to run")
parser.add_argument("--fname", type=str, default="ppo", metavar="P", help="policy to run")
parser.add_argument("--repeats", type=int, default=10, metavar="R", help="how many attempts we want to record")
args = parser.parse_args()

def main():
    # initialize filepaths
    directory = os.getcwd()
    fp = directory + "/saved_policies/"+args.fname
    
    # create figure for animation function
    env_name = args.env+"-v0"
    env = gym.make(env_name)
    agent = utils.load(fp)
    batch_rwd = 0
    for k in range(1, args.repeats+1):
        state = torch.Tensor(env.reset())
        env.render()
        done = False
        running_reward = 0
        while not done:
            action, _ = agent.select_action(state)
            state, reward, done, _  = env.step(action.detach().cpu().numpy())
            running_reward += reward
            state = torch.Tensor(state)
            env.render()
            env.record()
        batch_rwd = (batch_rwd*(k-1)+running_reward)/k
    print("Mean reward: {:.3f}".format(batch_rwd))
    
    
if __name__ == "__main__":
    main()