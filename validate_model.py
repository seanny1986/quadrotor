import torch
import quad
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("seaborn-deep")


cuda = True

def main():
    
    print("=> Loading model.pth.tar")
    dyn = torch.load("model.pth.tar")

    print("initializing aircraft")
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
    
    print("getting state")
    xyz, zeta, uvw, pqr = iris.get_state()
    action = trim+50

    xyz_nn = xyz.reshape((1,-1))
    zeta_nn = zeta.reshape((1,-1))
    uvw_nn = uvw.reshape((1,-1))
    pqr_nn = pqr.reshape((1,-1))
    action_nn = action.reshape((1,-1))

    xyz_nn = torch.from_numpy(xyz_nn).float()
    zeta_nn = torch.from_numpy(zeta_nn).float()#/dyn.zeta_norm
    uvw_nn = torch.from_numpy(uvw_nn).float()#/dyn.uvw_norm
    pqr_nn = torch.from_numpy(pqr_nn).float()#/dyn.pqr_norm
    action_nn = torch.from_numpy(action_nn).float()#/dyn.action_norm
    
    if cuda:
        xyz_nn = xyz_nn.cuda()
        zeta_nn = zeta_nn.cuda()
        uvw_nn = uvw_nn.cuda()
        pqr_nn = pqr_nn.cuda()
        action_nn = action_nn.cuda()

    data_nn = []
    data_actual = []
    time = []
    for i in range(steps):
        state = torch.cat([zeta_nn.sin(), zeta_nn.cos(), uvw_nn, pqr_nn],dim=1)
        state_action = torch.cat([state, action_nn],dim=1)
        xyz_nn, zeta_nn, uvw_nn, pqr_nn = dyn.transition(xyz_nn, state_action, dt)
        xyz, zeta, uvw, pqr, _, _, _, _ = iris.step(action)
        data_nn.append(xyz_nn.tolist()[0])
        data_actual.append(xyz.reshape((1,-1)).tolist()[0])
        time.append(i*dt)
        #zeta_nn = zeta_nn/dyn.zeta_norm
        #uvw_nn = uvw_nn/dyn.uvw_norm
        #pqr_nn = pqr_nn/dyn.pqr_norm
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312)
    ax3 = fig1.add_subplot(313)
    ax1.set_title("Position Error")
    ax3.set_xlabel("Time (s)")
    ax1.set_ylabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax3.set_ylabel("Z (m)")
    ax1.set_ylim([-3, 3])
    ax2.set_ylim([-3, 3])
    ax3.set_ylim([-3, 3])
    fig1.subplots_adjust(hspace=0.3)
    fig1.subplots_adjust(wspace=0.3)

    x_nn, y_nn, z_nn = [x[0] for x in data_nn], [x[1] for x in data_nn], [x[2] for x in data_nn]
    x_actual, y_actual, z_actual = [x[0] for x in data_actual], [x[1] for x in data_actual], [x[2] for x in data_actual]
    p5, p6 = ax1.plot(time, x_actual, time, x_nn)
    ax2.plot(time, y_actual, time, y_nn)
    ax3.plot(time, z_actual, time, z_nn)
    fig1.legend((p5, p6), ('Actual', 'Predicted'))
    plt.show()
    print("Saving figure")
    fig1.savefig('position_error.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()