import trainers.cem as cem_trainer
import trainers.ddpg as ddpg_trainer
import trainers.fmis as fmis_trainer
import trainers.gae as gae_trainer
import trainers.ppo as ppo_trainer
import trainers.qprop as qprop_trainer
import trainers.config as cfg

import threading

env_name = cfg.exp["env"]
algs = cfg.exp["algs"]

def main(env_name, algs):
    for alg in algs:
        make(env_name, alg)

    # collect data from threads and save for plotting

def make(env_name, alg):
    # return an instance of the trainer running on its own thread

    if alg == "cem":
        params = cfg.cem
        return cem_trainer.Trainer(env_name, params)
    if alg == "ddpg":
        params = cfg.ddpg
        return ddpg_trainer.Trainer(env_name, params)
    if alg == "fmis":
        params = cfg.fmis
        return fmis_trainer.Trainer(env_name, params)
    if alg == "gae":
        params = cfg.gae
        return gae_trainer.Trainer(env_name, params)
    if alg == "ppo":
        params = cfg.ppo
        return ppo_trainer.Trainer(env_name, params)
    if alg == "qprop":
        params = cfg.qprop
        return qprop_trainer.Trainer(env_name, params)

if __name__ == "__main__":
    main(env_name, algs)
    