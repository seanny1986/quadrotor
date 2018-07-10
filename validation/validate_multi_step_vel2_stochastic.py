import torch
import math
import numpy as np
import environments.envs as envs
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("seaborn-deep")

cuda = True
if cuda:
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.nn.Tensor

def main():

    print("=> Loading stochastic_multi_step.pth.tar")
    dyn = torch.load("/home/seanny/quadrotor/models/stochastic_multi_step.pth.tar")

    print("=> Initializing environment")
    env = envs.make("train_model")
    trim = env.iris.hover_rpm
    data_nn = []
    data_actual = []
    time = []
    H = 2
    ctrl_dt = 0.05
    steps = int(H/ctrl_dt)
    increment = np.array([0., 0., 0., 0.25])
    counter = 0
    frames = 100
    
    state = env.reset()
    action = trim
    args = None
    for i in range(steps):
        state = Tensor(state)
        sim_action = Tensor(action.copy()).unsqueeze(0)
        if cuda:
            sim_action = sim_action.cuda()
        pred_next_state, args = dyn.step(torch.cat([state, sim_action],dim=1), args)
        data_nn.append(pred_next_state)
        data_actual.append(state)
        state, _, _, _ = env.step(action)
        action += increment
    
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
    print("=> Saving figure as stochastic_multi_step_position_error.pdf")
    fig1.savefig('/home/seanny/quadrotor/figures/stochastic_multi_step_position_error.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()