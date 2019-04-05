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
plt.rc('text', usetex=True)
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)
#plt.rc('font', family='serif')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env', type=str, default=None, metavar='E', help='environment to plot')
parser.add_argument('--name', type=str, default='fig', metavar='N', help='name to save figure as')
args = parser.parse_args()

curr_dir = os.getcwd()

#### plot comparison
fp = curr_dir + "/data/RandomWaypointComparison/data.xlsx"

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Iterations")
ax.set_ylabel("Reward")
#fig.subplots_adjust(hspace=0.3)
#fig.subplots_adjust(wspace=0.3)

# loop through data folder and plot data
data = pd.read_excel(fp)
baseline = data[["Baseline-1","Baseline-2","Baseline-3","Baseline-4","Baseline-5"]]
term = data[["Term-1","Term-2","Term-3","Term-4","Term-5"]]

# join dataframes
data["mean_baseline"] = baseline.mean(axis=1)
data["std_baseline"] = baseline.std(axis=1)
data["mean_term"] = term.mean(axis=1)
data["std_term"] = term.std(axis=1)

xs = data["episode"]
ax.plot([],[])
ax.plot([],[])
ax.plot([],[])
ax.plot([],[])
ax.plot([],[])
ax.plot([],[])
ax.plot([],[])
ax.plot(xs, data["mean_baseline"], label="Baseline-10")
ax.plot(xs, data["mean_term"], label="Term-1")
ax.fill_between([],[])
ax.fill_between([],[])
ax.fill_between([],[])
ax.fill_between([],[])
ax.fill_between([],[])
ax.fill_between([],[])
ax.fill_between([],[])
ax.fill_between(xs, data["mean_baseline"]-data["std_baseline"], data["mean_baseline"]+data["std_baseline"], alpha=0.3)
ax.fill_between(xs, data["mean_term"]-data["std_term"], data["mean_term"]+data["std_term"], alpha=0.3)
plt.legend()         
plt.show()
fig.savefig(curr_dir + "/figures/comparison.pdf", bbox_inches="tight")
print("Figure saved")

####
####
####
#### plot term versus fixed!
fp = curr_dir + "/data/TrajectoryTerm/data.xlsx"

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Iterations")
ax.set_ylabel("Reward")
#fig.subplots_adjust(hspace=0.3)
#fig.subplots_adjust(wspace=0.3)

# loop through data folder and plot data
data = pd.read_excel(fp)
baseline_05 = data[["Baseline-05-1","Baseline-05-2","Baseline-05-3","Baseline-05-4","Baseline-05-5"]]
baseline_10 = data[["Baseline-10-1","Baseline-10-2","Baseline-10-3","Baseline-10-4","Baseline-10-5"]]
baseline_15 = data[["Baseline-15-1","Baseline-15-2","Baseline-15-3","Baseline-15-4","Baseline-15-5"]]
term_1 = data[["Obs-1-Run-1","Obs-1-Run-2","Obs-1-Run-3","Obs-1-Run-4","Obs-1-Run-5"]]
term_2 = data[["Obs-2-Run-1","Obs-2-Run-2","Obs-2-Run-3","Obs-2-Run-4","Obs-2-Run-5"]]
term_3 = data[["Obs-3-Run-1","Obs-3-Run-2","Obs-3-Run-3","Obs-3-Run-4","Obs-3-Run-5"]]

# join dataframes
data["mean_baseline_05"] = baseline_05.mean(axis=1)
data["std_baseline_05"] = baseline_05.std(axis=1)
data["mean_baseline_10"] = baseline_10.mean(axis=1)
data["std_baseline_10"] = baseline_10.std(axis=1)
data["mean_baseline_15"] = baseline_15.mean(axis=1)
data["std_baseline_15"] = baseline_15.std(axis=1)
data["mean_term_1"] = term_1.mean(axis=1)
data["std_term_1"] = term_1.std(axis=1)
data["mean_term_2"] = term_2.mean(axis=1)
data["std_term_2"] = term_2.std(axis=1)
data["mean_term_3"] = term_3.mean(axis=1)
data["std_term_3"] = term_3.std(axis=1)

xs = data["episode"]
ax.plot(xs, data["mean_baseline_05"], label="Baseline-05")
ax.plot(xs, data["mean_baseline_10"], label="Baseline-10")
ax.plot(xs, data["mean_baseline_15"], label="Baseline-15")
ax.plot([],[])
ax.plot(xs, data["mean_term_1"], label="Term-1")
ax.plot(xs, data["mean_term_2"], label="Term-2")
ax.plot(xs, data["mean_term_3"], label="Term-3")
ax.fill_between(xs, data["mean_baseline_05"]-data["std_baseline_05"], data["mean_baseline_05"]+data["std_baseline_05"], alpha=0.3)
ax.fill_between(xs, data["mean_baseline_10"]-data["std_baseline_10"], data["mean_baseline_10"]+data["std_baseline_10"], alpha=0.3)
ax.fill_between(xs, data["mean_baseline_15"]-data["std_baseline_15"], data["mean_baseline_15"]+data["std_baseline_15"], alpha=0.3)
ax.fill_between([],[])
ax.fill_between(xs, data["mean_term_1"]-data["std_term_1"], data["mean_term_1"]+data["std_term_1"], alpha=0.3)
ax.fill_between(xs, data["mean_term_2"]-data["std_term_2"], data["mean_term_2"]+data["std_term_2"], alpha=0.3)
ax.fill_between(xs, data["mean_term_3"]-data["std_term_3"], data["mean_term_3"]+data["std_term_3"], alpha=0.3)
plt.legend()         
plt.show()
fig.savefig(curr_dir + "/figures/term_vs_fixed.pdf", bbox_inches="tight")
print("Figure saved")


####
####
####
#### plot radius effect!
fp = curr_dir + "/data/Trajectory-v0/data.xlsx"

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Iterations")
ax.set_ylabel("Reward")
#fig.subplots_adjust(hspace=0.3)
#fig.subplots_adjust(wspace=0.3)

# loop through data folder and plot data
data = pd.read_excel(fp)
rad_05 = data[["Rad-05-1","Rad-05-2","Rad-05-3","Rad-05-4","Rad-05-5"]]
rad_10 = data[["Rad-10-1","Rad-10-2","Rad-10-3","Rad-10-4","Rad-10-5"]]
rad_15 = data[["Rad-15-1","Rad-15-2","Rad-15-3","Rad-15-4","Rad-15-5"]]
rad_20 = data[["Rad-20-1","Rad-20-2","Rad-20-3","Rad-20-4","Rad-20-5"]]

# join dataframes
data["mean_rad_05"] = rad_05.mean(axis=1)
data["std_rad_05"] = rad_05.std(axis=1)
data["mean_rad_10"] = rad_10.mean(axis=1)
data["std_rad_10"] = rad_10.std(axis=1)
data["mean_rad_15"] = rad_15.mean(axis=1)
data["std_rad_15"] = rad_15.std(axis=1)
data["mean_rad_20"] = rad_20.mean(axis=1)
data["std_rad_20"] = rad_20.std(axis=1)

xs = data["episode"]
ax.plot(xs, data["mean_rad_05"], label="Baseline-05")
ax.plot(xs, data["mean_rad_10"], label="Baseline-10")
ax.plot(xs, data["mean_rad_15"], label="Baseline-15")
ax.plot(xs, data["mean_rad_20"], label="Baseline-20")
ax.fill_between(xs, data["mean_rad_05"]-data["std_rad_05"], data["mean_rad_05"]+data["std_rad_05"], alpha=0.3)
ax.fill_between(xs, data["mean_rad_10"]-data["std_rad_10"], data["mean_rad_10"]+data["std_rad_10"], alpha=0.3)
ax.fill_between(xs, data["mean_rad_15"]-data["std_rad_15"], data["mean_rad_15"]+data["std_rad_15"], alpha=0.3)
ax.fill_between(xs, data["mean_rad_20"]-data["std_rad_20"], data["mean_rad_20"]+data["std_rad_20"], alpha=0.3)
plt.legend()         
plt.show()
fig.savefig(curr_dir + "/figures/goal_size_plot.pdf", bbox_inches="tight")
print("Figure saved")
