import trainers.config as cfg
import threading

"""
    Main experiment script. 
"""

env_name = cfg.exp["env"]
algs = cfg.exp["algs"]

def main(env_name, algs):
    for alg in algs:
        trainer = make(env_name, alg)
        trainer.train()

    # collect data from threads and save for plotting

def make(env_name, alg):
    """
        Builds and return an instance of a trainer for a given algorithm.
    """

    if alg == "cem":
        params = cfg.cem
        import trainers.cem as cem_trainer
        return cem_trainer.Trainer(env_name, params)
    if alg == "ddpg":
        params = cfg.ddpg
        import trainers.ddpg as ddpg_trainer
        return ddpg_trainer.Trainer(env_name, params)
    if alg == "fmis":
        params = cfg.fmis
        import trainers.fmis as fmis_trainer
        return fmis_trainer.Trainer(env_name, params)
    if alg == "gae":
        params = cfg.gae
        import trainers.gae as gae_trainer
        return gae_trainer.Trainer(env_name, params)
    if alg == "ppo":
        params = cfg.ppo
        import trainers.ppo as ppo_trainer
        return ppo_trainer.Trainer(env_name, params)
    if alg == "qprop":
        params = cfg.qprop
        import trainers.qprop as qprop_trainer
        return qprop_trainer.Trainer(env_name, params)

if __name__ == "__main__":
    main(env_name, algs)
    