import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import argparse
from math import sin, cos, pi
from matplotlib import rc

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

"""
    Function to play back saved policies and save video. I hate matplotlib.

    -- Sean Morrison, 2018
"""
parser = argparse.ArgumentParser(description="PyTorch actor-critic example")
parser.add_argument('--name', type=str, default='fig', metavar='N', help='name to save figure as')
args = parser.parse_args()

fig = plt.figure()
ax = fig.add_subplot(111)

# generate plot points for rotors
n = 35          # number of points for plotting rotors
r = 0.1         # propeller radius. This is cosmetic only
l = 0.15        # distance from COM to COT, cosmetic only

# generate circular points
p1 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r] for x in range(n+1)])
p2 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r] for x in range(n+1)])
p3 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r] for x in range(n+1)])
p4 = np.array([[cos(2*pi/n*x)*r, sin(2*pi/n*x)*r] for x in range(n+1)])

# shift to position
p1 += np.array([[l, 0.] for x in range(n+1)])
p2 += np.array([[0., -l] for x in range(n+1)])
p3 += np.array([[-l, 0.] for x in range(n+1)])
p4 += np.array([[0., l] for x in range(n+1)])

a1 = np.array([[(l-r)*(x/(n+1)), 0.] for x in range(n+1)])
a2 = np.array([[0., -(l-r)*(x/(n+1))] for x in range(n+1)])
a3 = np.array([[-(l-r)*(x/(n+1)), 0.] for x in range(n+1)])
a4 = np.array([[0., (l-r)*(x/(n+1))] for x in range(n+1)])

# plot rotors and arms
ax.plot(p1[:,0], p1[:,1], 'k')
ax.plot(p2[:,0], p2[:,1], 'k')
ax.plot(p3[:,0], p3[:,1], 'k')
ax.plot(p4[:,0], p4[:,1], 'k')

ax.plot(a1[:,0], a1[:,1], 'k')
ax.plot(a2[:,0], a2[:,1], 'k')
ax.plot(a3[:,0], a3[:,1], 'k')
ax.plot(a4[:,0], a4[:,1], 'k')

ax.scatter(l, 0., s=1000, marker=r"$\circlearrowright$",color="black")
ax.scatter(0., -l, s=1000, marker=r"$\circlearrowleft$",color="black")
ax.scatter(-l, 0., s=1000, marker=r"$\circlearrowright$",color="black")
ax.scatter(0., l, s=1000, marker=r"$\circlearrowleft$",color="black")

ax.text(l+0.15, 0., r"$m_{1}$", None)
ax.text(0., -l-0.15, r"$m_{2}$", None)
ax.text(-l-0.15, 0., r"$m_{3}$", None)
ax.text(0., l+0.15, r"$m_{4}$", None)

#ax.quiver()

ax.set_xlim(-0.4, 0.4)
ax.set_ylim(-0.4, 0.4)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.axis('off')
plt.show()
fig.savefig(os.getcwd() + "/figures/"+args.name+".pdf", bbox_inches="tight")