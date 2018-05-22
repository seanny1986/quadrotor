from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl


def draw3d(ax, xyz, R, quadcolor):
    ax.scatter(xyz[0,0], xyz[1,0], xyz[2,0], color=quadcolor)
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[0,0], R[0,1], R[0,2], pivot='tail', color='red')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[1,0], R[1,1], R[1,2], pivot='tail', color='green')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[2,0], R[2,1], R[2,2], pivot='tail', color='blue')