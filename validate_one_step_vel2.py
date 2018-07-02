import torch
import simulation.quadrotor2 as quad
import simulation.animation as ani
import simulation.config as cfg
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("seaborn-deep")

cuda = True
if cuda:
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.nn.Tensor

def main():

    print("=> Loading one_step_vel2.pth.tar")
    dyn = torch.load("/home/seanny/quadrotor/models/one_step2.pth.tar")

    print("=> Initializing aircraft from config")
    params = cfg.params
    iris = quad.Quadrotor(params)
    hover_rpm = iris.hov_rpm
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])+50.
    dt = iris.dt
    H = 1.5
    steps = int(H/dt)
    vis = ani.Visualization(iris, 10, quaternion=True)

    print("HOVER RPM: ", trim)
    input("Press to continue")

    plt.close("all")
    plt.ion()
    fig = plt.figure(0)
    axis3d = fig.add_subplot(111, projection='3d')
    
    xyz, zeta, q, uvw, pqr = iris.get_state()
    action = trim

    xyz_nn = xyz.reshape((1,-1))
    zeta_nn = zeta.reshape((1,-1))
    q_nn = q.reshape((1,-1))
    uvw_nn = uvw.reshape((1,-1))
    pqr_nn = pqr.reshape((1,-1))
    action_nn = action.reshape((1,-1))

    xyz_nn = torch.from_numpy(xyz_nn).float()
    zeta_nn = torch.from_numpy(zeta_nn).float()
    q_nn = torch.from_numpy(q_nn).float()
    uvw_nn = torch.from_numpy(uvw_nn).float()
    pqr_nn = torch.from_numpy(pqr_nn).float()
    action_nn = torch.from_numpy(action_nn).float()
    
    if cuda:
        xyz_nn = xyz_nn.cuda()
        zeta_nn = zeta_nn.cuda()
        q_nn = q_nn.cuda()
        uvw_nn = uvw_nn.cuda()
        pqr_nn = pqr_nn.cuda()
        action_nn = action_nn.cuda()

    data_nn = []
    data_actual = []
    time = []

    increment = np.array([0., 0., 0., 0.25])
    counter = 0
    frames = 100
    for i in range(steps):
        data_nn.append(xyz_nn.tolist()[0])
        data_actual.append(xyz.reshape((1,-1)).tolist()[0])
        time.append(i*dt)
        state = torch.cat([zeta_nn.sin(), zeta_nn.cos(), uvw_nn, pqr_nn],dim=1)
        state_action = torch.cat([state, action_nn],dim=1)
        xyz_nn, zeta_nn, uvw_nn, pqr_nn = dyn.transition(xyz_nn, q_nn, state_action, dt)
        xyz, zeta, q, uvw, pqr = iris.step(action)
        q_nn = q_nn = q.reshape((1,-1))
        q_nn = torch.from_numpy(q_nn).float()
        if cuda:
            q_nn = q_nn.cuda()
        if counter%frames == 0:
            plt.figure(0)
            axis3d.cla()
            vis.draw3d_quat(axis3d)
            axis3d.set_xlim(-3, 3)
            axis3d.set_ylim(-3, 3)
            axis3d.set_zlim(0, 6)
            axis3d.set_xlabel('West/East [m]')
            axis3d.set_ylabel('South/North [m]')
            axis3d.set_zlabel('Down/Up [m]')
            axis3d.set_title("Time %.3f s" %(i*dt))
            plt.pause(0.001)
            plt.draw()
        action += increment
        action_nn = torch.from_numpy(action).float()
        action_nn = action_nn.reshape((1,-1))
        if cuda:
            action_nn = action_nn.cuda()

    
    print("=> Plotting trajectory")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312)
    ax3 = fig1.add_subplot(313)
    ax1.set_title("Position Error")
    ax3.set_xlabel("Time (s)")
    ax1.set_ylabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax3.set_ylabel("Z (m)")
    ax1.set_ylim([-3.5, 3.5])
    ax2.set_ylim([-3.5, 3.5])
    ax3.set_ylim([-3.5, 3.5])
    fig1.subplots_adjust(hspace=0.3)
    fig1.subplots_adjust(wspace=0.3)

    x_nn, y_nn, z_nn = [x[0] for x in data_nn], [x[1] for x in data_nn], [x[2] for x in data_nn]
    x_actual, y_actual, z_actual = [x[0] for x in data_actual], [x[1] for x in data_actual], [x[2] for x in data_actual]
    p5, p6 = ax1.plot(time, x_actual, time, x_nn)
    ax2.plot(time, y_actual, time, y_nn)
    ax3.plot(time, z_actual, time, z_nn)
    fig1.legend((p5, p6), ('Actual', 'Predicted'))
    plt.show()
    input("Paused")
    print("=> Saving figure as position_error.pdf")
    fig1.savefig('/home/seanny/quadrotor/figures/position_error.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()