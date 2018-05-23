import quad
import model
import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import torch

style.use("seaborn-deep")

def main():

    epochs = 30000
    state_dim = 12
    action_dim = 4
    hidden_dim = 128
    dyn = model.Transition(state_dim, action_dim, hidden_dim, True)

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

    hover_thrust = (mass*9.81)/4.0
    hover_rpm = math.sqrt(hover_thrust/kt)
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    iris = quad.Quadrotor(mass, l, Jxx, Jyy, Jzz, kt, kq, kd1, kd2, dt)

    print("HOVER RPM: ", trim)
    input("Press to continue")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Linear Acceleration Loss")
    ax1.set_xlabel(r"Iterations $\times 10^2$")
    ax1.set_ylabel("Loss")
    fig1.subplots_adjust(hspace=0.3)
    fig1.subplots_adjust(wspace=0.3)
    fig1.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Angular Acceleration Loss")
    ax2.set_xlabel(r"Iterations $\times 10^2$")
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
        zeta_rand = np.random.uniform(low=-pi,high=pi,size=(3,1))
        uvw_rand = np.random.uniform(low=-20, high=20, size=(3,1))
        pqr_rand = np.random.uniform(low=-6, high=6, size=(3,1))

        # set random state
        iris.set_state(xyz_rand, zeta_rand, uvw_rand, pqr_rand)

        # generate random action, assume hover at 50%
        action = np.random.uniform(low=0, high=500, size=(4,))
        _, _, _, _, _, _, uvw_dot, pqr_dot = iris.step(action)

        # update network
        v_dot_loss, w_dot_loss = dyn.update(zeta_rand, uvw_rand, pqr_rand, action, uvw_dot, pqr_dot)

        if len(av_vdot)>10:
            del av_vdot[0]
            av_vdot.append(v_dot_loss)
        else:
            av_vdot.append(v_dot_loss)

        if len(av_wdot)>10:
            del av_wdot[0]
            av_wdot.append(w_dot_loss)
        else:
            av_wdot.append(w_dot_loss)

        average_vdot = float(sum(av_vdot))/float(len(av_vdot))
        average_wdot = float(sum(av_wdot))/float(len(av_wdot))
        data_vdot.append(average_vdot)
        data_wdot.append(average_wdot)
        iterations.append(counter)
        
        if counter%100 == 0:
            ax1.clear()
            ax1.plot(iterations,data_vdot)
            ax1.set_title("Linear Acceleration Loss")
            ax1.set_xlabel(r"Iterations $\times 10^2$")
            ax1.set_ylabel("Loss")
            fig1.canvas.draw()

            ax2.clear()
            ax2.plot(iterations,data_wdot)
            ax2.set_title("Angular Acceleration Loss")
            ax2.set_xlabel(r"Iterations $\times 10^2$")
            ax2.set_ylabel("Loss")
            fig2.canvas.draw()

        print(v_dot_loss, w_dot_loss)

        if counter > epochs:
            running = False
            print("Saving figures")
            fig1.savefig('vdot_loss.pdf', bbox_inches='tight')
            fig2.savefig('wdot_loss.pdf', bbox_inches='tight')
            print("Saving model")
            torch.save(dyn, "model.pth.tar")

        counter += 1

if __name__ == "__main__":
    main()