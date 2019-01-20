import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
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
parser.add_argument("--env", type=str, default="TrajectoryLineThree-v0", metavar="E", help="environment to run")
parser.add_argument("--fname", type=str, default="ppo", metavar="P", help="policy to run")
parser.add_argument("--vid", type=bool, default=False, metavar="V", help="determines whether to record video or not")
parser.add_argument("--repeats", type=int, default=1, metavar="R", help="how many attempts we want to record")
args = parser.parse_args()

def main():
    # initialize filepaths
    directory = os.getcwd()
    fp = directory + "/saved_policies/trpo-full-spectrum-4-TrajectoryTermThree-v0-final.pth.tar"
    
    # create list to store state information over the flight. This is... doing it the hard way,
    # but the matplotlib animation class doesn't want to do this easily :/
    env = gym.make(args.env)
    agent = utils.load(fp)
    batch_rwd = 0
    distances = []
    accelerations = []
    for k in range(1, args.repeats+1):
        state = torch.Tensor(env.reset())
        #env.render()
        done = False
        hidden = None
        reward_sum = 0
        distance = []
        acceleration = []
        while not done:
            env.render()
            #env.record()
            action, _ = agent.select_action(state)
            term, _, hidden, _ = agent.terminate(state, hidden)
            next_state, reward, done, info = env.step(action.cpu().data.numpy(), term.cpu().item())
            
            dist = env.get_distance()
            lin_accel = info["lin_accel"]
            print("linear: ", lin_accel)
            print("angular: ", info["ang_accel"])
            
            distance.append(dist)
            acceleration.append(lin_accel)

            reward_sum += reward
            next_state = torch.Tensor(next_state)
            state = next_state
        distances.append(distance)
        accelerations.append(acceleration)
    print("Mean reward: {:.3f}".format(batch_rwd))    
    
    for k in range(args.repeats):
        x = distances[k]
        y = accelerations[k]
        plt.plot(x, y)
    plt.show()
    

if __name__ == "__main__":
    main()