"""
    This script contains training parameters for the policy search algorithms defined in policies.
    Each algorithm has its own training methodology, and thus its own set of parameters. If you
    implement a new policy search algorithm, you will need to put the policy architecture in
    the policies folder, and build a training wrapper in this folder. Then you will need to add
    the training parameters to this script so that they can be imported into your experiment script.

    -- Sean Morrison, 2018
"""

exp = { 
        "env": "flying_skills",
        "algs": ["fmis"]
        }

cem = { 
        "hidden_dim": 32,
        "iterations": 5000,
        "gamma": 0.99,
        "seed": 343,
        "log_interval": 10,
        "pop_size": 32,
        "elite_frac": 0.2,
        "sigma": 0.2,
        "render": True,
        "save": False,
        "cuda": True
        }

ddpg = {
        "hidden_dim": 32,
        "iterations": 5000,
        "gamma": 0.99,
        "mem_len": 1000000,
        "seed": 343,
        "log_interval": 10,
        "warmup": 50,
        "batch_size": 64,
        "render": True,
        "save": False,
        "cuda": True,
        "ou_scale": 1,
        "ou_mu": 0.75,
        "ou_sigma": 0.05
        }

fmis = {
        "hidden_dim": 32,
        "iterations": 5000,
        "gamma": 0.99,
        "seed": 343,
        "log_interval": 10,
        "render": True,
        "save": False,
        "cuda": True
        }

gae = {
        "hidden_dim": 64,
        "iterations": 5000,
        "gamma": 0.99,
        "seed": 343,
        "log_interval": 10,
        "render": False,
        "save": False,
        "cuda": True
        }

ppo = {
        "hidden_dim": 32,
        "iterations": 5000,
        "gamma": 0.99,
        "seed": 343,
        "log_interval": 10,
        "render": False,
        "save": False,
        "cuda": True
        }

qprop = {
        "hidden_dim": 32,
        "iterations": 5000,
        "gamma": 0.99,
        "mem_len": 1000000,
        "seed": 343,
        "log_interval": 10,
        "warmup": 50,
        "batch_size": 64,
        "render": False,
        "save": False,
        "cuda": True
        }

trpo = {}