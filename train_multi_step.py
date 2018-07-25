import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import torch
import torch.optim as optim
import torch.nn.functional as F
import utils
import numpy as np
import environments.envs as envs
import models.multi_step_vel2_stochastic as model

style.use("seaborn-deep")
GPU = True

def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

def generate_action(vector, n):
    param = 2
    K_ss = kernel(vector, vector, param)
    L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))
    action = np.dot(L, np.random.normal(size=(n,4)))
    return action

def main():
    epochs = 100000
    input_dim, hidden_dim, output_dim = 16, 32, 3
    dyn = model.Transition(input_dim, hidden_dim, output_dim, GPU)
    if GPU:
        dyn = dyn.cuda()
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.nn.Tensor

    env = envs.make("model_training")
    max_rpm = env.action_bound[1]

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
    optimizer = optim.Adam(dyn.parameters(),lr=1e-4)
    criterion = torch.nn.MSELoss(size_average=True)

    H = 20
    while running:
        
        # generate smooth random actions
        baseline = np.random.uniform(low=0.4*max_rpm, high=max_rpm)
        input_vector = np.linspace(-5, 5, H).reshape(-1,1)
        action_vector = generate_action(input_vector, H)

        # set random state
        state = Tensor(env.reset())      
        state_actions = []
        next_states = []
        
        # run trajectory
        for i in range(H):
            action = action_vector[i,:]+baseline
            action_tensor = torch.from_numpy(np.expand_dims(action,axis=0))
            if GPU:
                action_tensor = action_tensor.float().cuda()
            state_action = torch.cat([state, action_tensor],dim=1)
            next_state, _, _, _ = env.step(action)
            next_state = Tensor(next_state)
            state_actions.append(state_action)
            next_states.append(next_state)
            state = next_state
            
        loss = dyn.update(optimizer, criterion, state_actions, next_states)

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
            torch.save(dyn, "/home/seanny/quadrotor/models/stochastic_multi_step.pth.tar")

if __name__ == "__main__":
    main()