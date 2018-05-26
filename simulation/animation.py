from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
from math import sin, cos, pi
import numpy as np
import numpy.matlib

class Visualization:
    def __init__(self, aircraft, n):
        self.aircraft = aircraft
        self.r = aircraft.prop_radius
        self.l = aircraft.l
        self.n = n

        self.p1 = np.array([[cos(2*pi/n*x)*self.r, sin(2*pi/n*x)*self.r, 0] for x in range(n+1)])
        self.p2 = np.array([[cos(2*pi/n*x)*self.r, sin(2*pi/n*x)*self.r, 0] for x in range(n+1)])
        self.p3 = np.array([[cos(2*pi/n*x)*self.r, sin(2*pi/n*x)*self.r, 0] for x in range(n+1)])
        self.p4 = np.array([[cos(2*pi/n*x)*self.r, sin(2*pi/n*x)*self.r, 0] for x in range(n+1)])

        self.p1 += np.array([[self.l, 0.0, 0.0] for x in range(n+1)])
        self.p2 += np.array([[0.0, -self.l, 0.0] for x in range(n+1)])
        self.p3 += np.array([[-self.l, 0.0, 0.0] for x in range(n+1)])
        self.p4 += np.array([[0.0, self.l, 0.0] for x in range(n+1)])

    def draw3d(self, ax):
        xyz, R = self.aircraft.xyz, self.aircraft.R1(self.aircraft.zeta).T
        
        ax.scatter(xyz[0,0], xyz[1,0], xyz[2,0], color='black')
        ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[0,0], R[0,1], R[0,2], pivot='tail', color='red')
        ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[1,0], R[1,1], R[1,2], pivot='tail', color='green')
        ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[2,0], R[2,1], R[2,2], pivot='tail', color='blue')
       
        # rotate to aircraft attitude
        p1 = np.einsum('ij,kj->ik', self.p1, R.T)
        p2 = np.einsum('ij,kj->ik', self.p2, R.T)
        p3 = np.einsum('ij,kj->ik', self.p3, R.T)
        p4 = np.einsum('ij,kj->ik', self.p4, R.T)

        # shift to 
        p1 = np.matlib.repmat(xyz.T,self.n+1,1)+p1
        p2 = np.matlib.repmat(xyz.T,self.n+1,1)+p2
        p3 = np.matlib.repmat(xyz.T,self.n+1,1)+p3
        p4 = np.matlib.repmat(xyz.T,self.n+1,1)+p4

        # plot rotated 
        ax.plot(p1[:,0], p1[:,1], p1[:,2],'k')
        ax.plot(p2[:,0], p2[:,1], p2[:,2],'k')
        ax.plot(p3[:,0], p3[:,1], p3[:,2],'k')
        ax.plot(p4[:,0], p4[:,1], p4[:,2],'k')