import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.style as style
import os
import argparse

"""
    function to import data from saved CSV files and generate plots for papers.

    -- Sean Morrison, 2018
"""

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env', type=str, default=None, metavar='E', help='environment to plot')
parser.add_argument('--name', type=str, default='fig', metavar='N', help='name to save figure as')
args = parser.parse_args()

style.use("seaborn-deep")
curr_dir = os.getcwd()
directory = curr_dir + "/data/"+args.env+"-v0"

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(args.env + "-v0 Policy Return")
ax.set_xlabel("Episodes")
ax.set_ylabel("Reward")
fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.3)

# loop through data folder and plot data
ppo = []
gae = []
trpo = []
for f in os.listdir(directory):
    if f.endswith(".csv"):
        if "ppo" in f:
            print(os.path.join(directory, f))
            data = pd.read_csv(os.path.join(directory, f))
            ppo.append(data)
        elif "gae" in f:
            print(os.path.join(directory, f))
            data = pd.read_csv(os.path.join(directory, f))
            gae.append(data)
        elif "trpo" in f:
            data = pd.read_csv(os.path.join(directory, f))
            trpo.append(data)

# join dataframes
ppo = pd.concat(ppo, axis=1)
gae = pd.concat(gae, axis=1)
trpo = pd.concat(trpo, axis=1)
ppo_temp = ppo.drop(["episode"], axis=1)
gae_temp = gae.drop(["episode"], axis=1)
trpo_temp = trpo.drop(["episode"], axis=1)
ppo["mean"] = ppo_temp.mean(axis=1)
gae["mean"] = gae_temp.mean(axis=1)
trpo["mean"] = trpo_temp.mean(axis=1)
ppo["stdev"] = ppo_temp.std(axis=1)
gae["stdev"] = gae_temp.std(axis=1)
trpo["stdev"] = trpo_temp.std(axis=1)

data = [ppo, gae, trpo]
for i, d in enumerate(data):
    xs = d["episode"]
    ys = d["mean"]
    pos_std = d["mean"]+d["stdev"]
    neg_std = d["mean"]-d["stdev"]
    if i == 0: 
        name = "ppo" 
    elif i == 1: 
        name = "gae"
    elif i == 2:
        name = "trpo"
    ax.plot(xs, ys, label=name)
    ax.fill_between(xs, neg_std, pos_std, alpha=0.3)
plt.legend()         
plt.show()
fig.savefig(curr_dir + "/figures/"+args.name+".pdf", bbox_inches="tight")
print("Figure saved")
