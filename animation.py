from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl


def draw3d(ax, xyz, R, quadcolor):
    ax.scatter(xyz[0,0], xyz[1,0], xyz[2,0], color=quadcolor)
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[0,0], R[0,1], R[0,2], pivot='tail', \
            color='red')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[1,0], R[1,1], R[1,2], pivot='tail', \
            color='green')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[2,0], R[2,1], R[2,2], pivot='tail', \
            color='blue')

def draw_edges(fig, X, fc, n):
    agents = fc.agents
    edges =  fc.edges
    m = fc.m
    B = fc.B
    pl.figure(fig)
    a, b = 0, 0

    for i in range(0, edges):
        for j in range(0, agents):
            if B[j,i] == 1:
                a = j
            elif B[j,i] == -1:
                b = j

        if m == 2:
            if i == n:
                pl.plot([X[m*a], X[m*b]], [X[m*a+1], X[m*b+1]], 'r--', lw=2)
            else:
                pl.plot([X[m*a], X[m*b]], [X[m*a+1], X[m*b+1]], 'k--', lw=2)