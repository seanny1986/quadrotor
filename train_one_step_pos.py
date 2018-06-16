import simulation.quadrotor as quad
import simulation.config as cfg
import models.one_step_pos as model
import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import torch

style.use("seaborn-deep")

def main():

    epochs = 250000
    state_dim = 9
    action_dim = 4
    hidden_dim = 64
    dyn = model.Transition(state_dim, action_dim, hidden_dim, True)

    params = cfg.params
    iris = quad.Quadrotor(params)
    hover_rpm = iris.hov_rpm
    max_rpm = iris.max_rpm
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])

    print("HOVER RPM: ", trim)
    print("Terminal Velocity: ", iris.terminal_velocity)
    print("Terminal Rotation: ", iris.terminal_rotation)
    input("Press to continue")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Position Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    fig1.subplots_adjust(hspace=0.3)
    fig1.subplots_adjust(wspace=0.3)
    fig1.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Attitude Loss")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Loss")
    fig2.subplots_adjust(hspace=0.3)
    fig2.subplots_adjust(wspace=0.3)
    fig2.show()
    
    av_vdot = []
    av_wdot = []
    data_vdot = []
    data_wdot = []
    iterations = []
    counter = 0

    running = True
    while running:
        
        # generate random state
        xyz_rand = np.random.uniform(low=-15, high=15, size=(3,1))
        zeta_rand = np.random.uniform(low=-2*pi,high=2*pi,size=(3,1))
        uvw_rand = np.random.uniform(low=-iris.terminal_velocity, high=iris.terminal_velocity, size=(3,1))
        pqr_rand = np.random.uniform(low=-iris.terminal_rotation, high=iris.terminal_rotation, size=(3,1))

        # set random state
        iris.set_state(xyz_rand, zeta_rand, uvw_rand, pqr_rand)

        # generate random action, assume hover at 50%
        action = np.random.uniform(low=0, high=max_rpm, size=(4,))
        xyz, zeta, uvw, pqr = iris.step(action)

        # update network
        xyz_loss, zeta_loss = dyn.update(xyz_rand, zeta_rand, action, xyz, zeta)

        if len(av_vdot)>10:
            del av_vdot[0]
            av_vdot.append(xyz_loss)
        else:
            av_vdot.append(xyz_loss)

        if len(av_wdot)>10:
            del av_wdot[0]
            av_wdot.append(zeta_loss)
        else:
            av_wdot.append(zeta_loss)

        average_vdot = float(sum(av_vdot))/float(len(av_vdot))
        average_wdot = float(sum(av_wdot))/float(len(av_wdot))
        
        if counter%100 == 0:
            data_vdot.append(average_vdot)
            data_wdot.append(average_wdot)
            iterations.append(counter/100.)
            ax1.clear()
            ax1.plot(iterations,data_vdot)
            ax1.set_title("Position Loss")
            ax1.set_xlabel(r"Iterations $\times 10^{2}$")
            ax1.set_ylabel("Loss")
            fig1.canvas.draw()

            ax2.clear()
            ax2.plot(iterations,data_wdot)
            ax2.set_title("Attitude Loss")
            ax2.set_xlabel(r"Iterations $\times 10^{2}$")
            ax2.set_ylabel("Loss")
            fig2.canvas.draw()
        counter += 1

        print(xyz_loss, zeta_loss)

        if counter > epochs:
            running = False
            print("Saving figures")
            fig1.savefig('xya_loss.pdf', bbox_inches='tight')
            fig2.savefig('zeta_loss.pdf', bbox_inches='tight')
            print("Saving model")
            torch.save(dyn, "/home/seanny/quadrotor/models/one_step_pos.pth.tar")

        

if __name__ == "__main__":
    main()