import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.trajectory_gpt2 import GPT2Model


from kmeans_pytorch import kmeans

import os

class MultiDecisionTransformer(DecisionTransformer):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    Modified for handling multi environments
    """

    def __init__(
        self,
        args,
        envs,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        self.act_dims = [env.action_space.shape[0] for env in envs]
        self.state_dims = [env.observation_space.shape[0] for env in envs]
        self.env_ids = [env.env.spec.id for env in envs]
        self.index_to_id = {i: env_id for i, env_id in enumerate(self.env_ids)}
        self.n_envs = len(envs)

        self.env_id = None
        self.env_index = None
        #import pdb; pdb.set_trace()
        super().__init__(
            args,
            self.state_dims[0], # set to temp value
            self.act_dims[0],
            hidden_size,
            max_length=max_length,
            max_ep_len=max_ep_len,
            action_tanh=action_tanh,
            **kwargs
        )

        # Create unique embeddings for each environment, and action predictions
    
        self.embed_returns = []
        self.embed_states = []
        self.embed_actions = []
        self.embed_lns = []
        for i in range(len(envs)):
            self.embed_returns.append(nn.Linear(1, hidden_size))
            self.embed_states.append(nn.Linear(self.state_dims[i], self.hidden_size))
            self.embed_actions.append(nn.Linear(self.act_dims[i], self.hidden_size))

            self.embed_lns.append(nn.LayerNorm(self.hidden_size))
        
        self.predict_actions = []
        for i in range(len(envs)):
            self.predict_actions.append(
                nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dims[i])]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            ))
        # if a finetuned model on a certain environment
        self.specializations = []

        self.transformer_pre = None # Weights at start of training
        self.transformer_post = None # Current transformer weights
        
    def toggle_env(self, env_id):
        self.env_id = env_id
        self.env_index = self.env_ids.index(env_id)

        self.state_dim = self.state_dims[self.env_index]
        self.act_dim = self.act_dims[self.env_index]

        self.embed_return = self.embed_returns[self.env_index]
        self.embed_state = self.embed_states[self.env_index]
        self.embed_action = self.embed_actions[self.env_index]
        self.embed_ln = self.embed_lns[self.env_index]

        self.predict_action = self.predict_actions[self.env_index]