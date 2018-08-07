"""
    This script contains training parameters for the policy search algorithms defined in policies.
    Each algorithm has its own training methodology, and thus its own set of parameters. If you
    implement a new policy search algorithm, you will need to put the policy architecture in
    the policies folder, and build a training wrapper in this folder. Then you will need to add
    the training parameters to this script so that they can be imported into your experiment script.

    -- Sean Morrison, 2018
"""

exp = { 
        "env": "hover-v0",
        "algs": ["ppo"]
        }

cem = { 
        "hidden_dim": 32,
        "iterations": 10000,
        "gamma": 0.99,
        "lr": 1e-4,
        "seed": 343,
        "log_interval": 10,
        "pop_size": 32,
        "elite_frac": 0.2,
        "sigma": 0.2,
        "render": True,
        "save": False,
        "cuda": True,
        "logging": True
        }

ddpg = {
        "network_settings": {
                                "gamma": 0.99,
                                "tau": 0.01
                                },
        "hidden_dim": 64,
        "iterations": 10000,
        "mem_len": 1000000,
        "lr": 1e-5,
        "seed": 343,
        "log_interval": 10,
        "warmup": 50,
        "batch_size": 64,
        "ou_scale": 0.75,
        "ou_mu": 0.2,
        "ou_sigma": 0.15,
        "render": True,
        "save": False,
        "cuda": True,
        "logging": True
        }

gae = {
        "network_settings": {
                                "gamma": 0.995,
                                "lambda": 0.92
                                },
        "hidden_dim": 32,
        "iterations": 10000,
        "batch_size": 8,
        "epochs": 16,
        "lr": 1e-4,
        "seed": 343,
        "log_interval": 10,
        "render": True,
        "save": False,
        "cuda": True,
        "logging": False
        }

ppo = { 
        "network_settings": {
                                "gamma": 0.995,
                                "lambda": 0.97,
                                "eps": 0.15
                                },
        "hidden_dim": 64,
        "iterations": 5000,
        "batch_size": 512,
        "epochs":3,
        "lr": 5e-5,
        "seed": 343,
        "log_interval": 10,
        "render": False,
        "save": False,
        "cuda": True,
        "logging": True
        }

trpo = {
        "network_settings": {
                                "gamma": 0.995,
                                "tau": 0.97,
                                "l2_reg": 1e-3,
                                "max_kl": 1e-2,
                                "damping": 1e-1
                                },
        "hidden_dim": 64,
        "iterations": 5000,
        "log_interval": 10,
        "warmup": 50,
        "batch_size": 512,
        "seed": 343,
        "render": True,
        "save": False,
        "cuda": False,
        "logging": True
        }
        