import config as cfg
import multiprocessing as mp

"""
    Main experiment script. We pull the experiment config parameters from file and then fire up a bunch of
    instances, with each one running in its own thread (no communication between different instances). This 
    lets us train multiple algorithms on the same environment at the same time. The trainer logs all of the 
    data and saves it to the correct location, after which we can plot it all out using matplotlib or R. 

    -- Sean Morrison, 2018
"""

# grab experiment environment and list of algorithms
env_name = cfg.exp["env"]
algs = cfg.exp["algs"]
processes = []

# start training processes
def main(env_name, algs):
    try :
        for alg in algs:
            p = mp.Process(target=make, args=(env_name, alg))
            processes.append(p)
            p.start()

        # Wait for processes to finish
        [p.join() for p in processes]

    #Listens for ctrl+c input
    except KeyboardInterrupt:
        print("\n---Terminating all processes---")
        for p in processes:
            p.terminate()

def make(env_name, alg):
    """
        Builds and returns an instance of a trainer for a given algorithm. All of these algorithms have been
        validated on basic tasks and shown to learn. You can edit the training parameters in the config file
        found in the root folder. 
    """

    if alg == "cem":
        params = cfg.cem
        import algs.ind.cem as cem
        print("---Initializing CEM in env: "+env_name+"---")
        return cem.Trainer(env_name, params)
    if alg == "ddpg":
        params = cfg.ddpg
        import algs.ind.ddpg as ddpg
        print("---Initializing DDPG in env: "+env_name+"---")
        return ddpg.Trainer(env_name, params)
    if alg == "gae":
        params = cfg.gae
        import algs.ind.gae as gae
        print("---Initializing GAE in env: "+env_name+"---")
        return gae.Trainer(env_name, params)
    if alg == "ppo":
        params = cfg.ppo
        import algs.ind.ppo as ppo
        print("---Initializing PPO in env: "+env_name+"---")
        return ppo.Trainer(env_name, params)
    if alg == "ma_trpo":
        params = cfg.trpo
        import algs.ind.ma_trpo as ma_trpo
        print("---Initializing MA_TRPO in env: "+env_name+"---")
        return ma_trpo.Trainer(env_name, params)
    if alg == "trpo":
        params = cfg.trpo
        import algs.ind.trpo as trpo
        print("---Initializing TRPO in env: "+env_name+"---")
        return trpo.Trainer(env_name, params)
    if alg == "trpo-h":
        params = cfg.trpo
        import algs.ind.trpo_h as trpo_h
        print("---Initializing TRPO-H in env: "+env_name+"---")
        return trpo_h.Trainer(env_name, params)
    if alg == "trpo-peb":
        params = cfg.trpo
        import algs.ind.trpo_peb as trpo_peb
        print("---Initializing TRPO-PEB in env: "+env_name+"---")
        return trpo_peb.Trainer(env_name, params)
    if alg == "trpo-skip":
        params = cfg.trpo
        import algs.ind.trpo_skip as trpo_skip
        print("---Initializing TRPO-Skip in env: "+env_name+"---")
        return trpo_skip.Trainer(env_name, params)

if __name__ == "__main__":
    main(env_name, algs)
    