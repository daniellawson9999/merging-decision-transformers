import gym
import numpy as np

import collections
import pickle

import d4rl
import os


datasets = []

stat_dict = {}
for env_name in ["halfcheetah", "hopper", "walker2d"]:
    for dataset_type in ["medium", "medium-replay", "expert", "medium-expert"]:
        name = f"{env_name}-{dataset_type}-v2"
        with open(name + '.pkl', "rb") as f:
            trajectories = pickle.load(f)


        states = []
        for path in trajectories:
            states.append(path["observations"])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        stat_dict[name] = {}
        stat_dict[name]['state_mean'] = state_mean
        stat_dict[name]['state_std'] = state_std

pickle.dump(stat_dict, open('state_stats.pkl','wb'))

            
