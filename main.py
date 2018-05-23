import quad
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


plt.close("all")
fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')

mass = 0.65
l = 0.23
Jxx = 7.5e-3
Jyy = 7.5e-3
Jzz = 1.3e-2
kt = 3.13e-5
kq = 7.5e-7
kd1 = 9e-3
kd2 = 9e-4
dt = 0.05
T = 1.5

x, y, z = 0, 0, 0
phi, theta, psi = 0, 0, 0

iris = quad.Quadrotor(mass, l, Jxx, Jyy, Jzz, kt, kq, kd1, kd2, dt)

def main():
    time = np.linspace(0, T, T/dt)
    hover_thrust = (mass*9.81)/4.0
    hover_rpm = math.sqrt(hover_thrust/kt)
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    rpm = trim+50
    
    for t in time:
        xyz, zeta, _, _, _, _, _, _ = iris.step(rpm)
        x = xyz[0,0]
        y = xyz[1,0]
        z = xyz[2,0]
        phi = zeta[0,0]
        theta = zeta[1,0]
        psi = zeta[2,0]

        ani = animation.FuncAnimation(fig, animate, frames=300, interval=interval, blit=True, init_func=init)

def frames():
    while True:
        yield iris

def animate(args):
    ax.scatter(xyz[0,0], xyz[1,0], xyz[2,0], color='black')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[0,0], R[0,1], R[0,2], pivot='tail', color='red')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[1,0], R[1,1], R[1,2], pivot='tail', color='green')
    ax.quiver(xyz[0,0], xyz[1,0], xyz[2,0], R[2,0], R[2,1], R[2,2], pivot='tail', color='blue')

    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

if __name__ == "__main__":
    main()