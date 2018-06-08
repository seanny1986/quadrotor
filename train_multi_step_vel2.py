import simulation.quadrotor2 as quad
import simulation.config as cfg
import models.multi_step_vel2 as model
import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import torch
import torch.optim as optim
import torch.nn.functional as F
import utils

style.use("seaborn-deep")
GPU = True
def main():

    epochs = 100000
    input_dim, hidden_dim, output_dim = 16, 32, 3
    dyn = model.Transition(input_dim, hidden_dim, output_dim, GPU)
    if GPU:
        dyn = dyn.cuda()

    params = cfg.params
    iris = quad.Quadrotor(params)
    hover_rpm = iris.hov_rpm
    max_rpm = iris.max_rpm
    trim = np.array([hover_rpm, hover_rpm, hover_rpm, hover_rpm])
    dt = iris.dt

    print("HOVER RPM: ", trim)
    print("Terminal Velocity: ", iris.terminal_velocity)
    print("Terminal Rotation: ", iris.terminal_rotation)
    input("Press to continue")

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Linear Velocity Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    fig1.subplots_adjust(hspace=0.3)
    fig1.subplots_adjust(wspace=0.3)
    fig1.show()
    
    av = []
    data = []
    iterations = []
    counter = 0

    running = True
    trajectory_len = 10
    optimizer = optim.Adam(dyn.parameters(),lr=1e-7)
    criterion = torch.nn.MSELoss(size_average=True)
    while running:
        
        # generate random state
        xyz_rand = np.random.uniform(low=-15, high=15, size=(3,1))
        zeta_rand = np.random.uniform(low=-2*pi,high=2*pi,size=(3,1))
        uvw_rand = np.random.uniform(low=-iris.terminal_velocity, high=iris.terminal_velocity, size=(3,1))
        pqr_rand = np.random.uniform(low=-iris.terminal_rotation, high=iris.terminal_rotation, size=(3,1))

        print("XYZ: ", xyz_rand)
        print("ZETA: ", zeta_rand)
        print("UVW: ", uvw_rand)
        print("PQR: ", pqr_rand)

        input("Paused")

        # set random state
        iris.set_state(xyz_rand, zeta_rand, uvw_rand, pqr_rand)
                
        xyzs = []
        zetas = []
        uvws = []
        pqrs = []
        actions = []

        xyzs.append(xyz_rand)
        zetas.append(zeta_rand)
        uvws.append(uvw_rand)
        pqrs.append(pqr_rand)

        # run trajectory
        for i in range(trajectory_len):
            action = np.random.uniform(low=0, high=max_rpm, size=(4,))        
            xyz, zeta, _, uvw, pqr = iris.step(action)

            print("XYZ: ", xyz)
            print("ZETA: ", zeta)
            print("UVW: ", uvw)
            print("PQR: ", pqr)

            input("Paused")

            xyzs.append(xyz.copy())
            zetas.append(zeta.copy())
            uvws.append(uvw.copy())
            pqrs.append(pqr.copy())
            actions.append(action)
            
        loss = dyn.update(optimizer, criterion, xyzs, zetas, uvws, pqrs, actions)

        if len(av)>10:
            del av[0]
            av.append(loss)
        else:
            av.append(loss)

        average = float(sum(av))/float(len(av))
        
        if counter%100 == 0:
            data.append(average)
            iterations.append(counter/100.)
            ax1.clear()
            ax1.plot(iterations,data)
            ax1.set_title("Linear Velocity Loss")
            ax1.set_xlabel(r"Iterations $\times 10^{2}$")
            ax1.set_ylabel("Loss")
            fig1.canvas.draw()
        counter += 1

        print(loss)

        if counter > epochs:
            running = False
            print("Saving figures")
            fig1.savefig('multi_step_loss.pdf', bbox_inches='tight')
            print("Saving model")
            torch.save(dyn, "/home/seanny/quadrotor/models/multi_step.pth.tar")

        

if __name__ == "__main__":
    main()