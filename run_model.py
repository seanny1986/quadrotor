import torch
import os
import matplotlib.pyplot as plt
import environments.envs as envs
import numpy as np
import models.one_step as model


directory = os.getcwd()
fp = directory + "/saved_models/one_step.pth.tar"
dyn = torch.load(fp)
steps = 20
def main():
    env = envs.make("model_training")
    state = np.array(env.reset(),dtype="float32")
    trim = env.trim
    add = np.array([0. ,0. ,0., 0.5])
    action = np.array(trim,dtype="float32")
    for i in range(0, steps):
        s = torch.from_numpy(state.copy())
        a = torch.from_numpy(action.copy().reshape(1,-1))
        s_a = torch.cat([s, a], dim=1).float()
        xyz = dyn.pos(s_a)
        zeta = dyn.att(s_a)
        uvw = dyn.vel(s_a)
        pqr = dyn.ang(s_a)
        RPM = dyn.rpm(s_a)
        ns = torch.cat([xyz, zeta, uvw, pqr, RPM],dim=1)
        state = np.array(env.step(action)[0],dtype="float32")
        action += add
    print("delta: ", ns-torch.from_numpy(state))
    print(torch.atan2(ns[:,3:6],ns[:,6:9]).detach().numpy())
    print(np.arctan2(state[:,3:6],state[:,6:9]))

if __name__ == "__main__":
    main()