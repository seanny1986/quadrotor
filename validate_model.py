import torch
import quad
import math
import numpy as np



def main():
    
    dyn = torch.load("model.pth.tar")

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
    H = 1.0

    steps = int(H/dt)

    hover_thrust = (mass*9.81)/4.0
    hover_rpm = math.sqrt(hover_thrust/kt)
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    iris = quad.Quadrotor(mass, l, Jxx, Jyy, Jzz, kt, kq, kd1, kd2, dt)

    print("HOVER RPM: ", trim)
    input("Press to continue")
    
    xyz, zeta, uvw, pqr = iris.get_state()
    action = trim+50
    for i in range(steps):
        xyz, zeta, uvw, pqr, xyz_dot, zeta_dot, uvw_dot, pqr_dot = iris.step(action)



if __name__ == "__main__":
    main()