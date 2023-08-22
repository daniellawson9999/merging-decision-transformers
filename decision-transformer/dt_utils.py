import copy
import pickle
import random
import os
from typing import Union
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import gym
from tqdm import tqdm

from torch import optim
from itertools import chain


env_config = {
    'Hopper-v3': {
        'target_return': 3600,
            'expert': 
            {
            'state_mean': np.array([ 1.3490015, -0.11208222, -0.5506444, -0.13188992, -0.00378754, 2.6071432, 0.02322114, -0.01626922, -0.06840388, -0.05183131, 0.04272673]),
            'state_std': np.array([0.15980862, 0.0446214, 0.14307782, 0.17629202, 0.5912333, 0.5899924, 1.5405099, 0.8152689, 2.0173461, 2.4107876, 5.8440027 ])
            },
            'short': 'hopper'
    },
    'HalfCheetah-v3': {
        'target_return': 12000,
        'expert': 
             {
                'state_mean': np.array([ -0.04489148, 0.03232588, 0.06034835, -0.17081226, -0.19480659, -0.05751596, 0.09701628, 0.03239211, 11.047426, -0.07997331, -0.32363534, 0.36297753, 0.42322603, 0.40836546, 1.1085187, -0.4874403, -0.0737481 ]),
                'state_std': np.array([0.04002118, 0.4107858, 0.54217845, 0.41522816, 0.23796624, 0.62036866, 0.30100912, 0.21737163, 2.2105937, 0.572586, 1.7255033, 11.844218, 12.06324, 7.0495934, 13.499867, 7.195647, 5.0264325])
             },
        'short': 'halfcheetah'
    },
    'Walker2d-v3': {
        'target_return': 5000,
        'short': 'walker2d'
    }

}

def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # we don't care about the past rewards in this model

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    if model.config.max_length is not None:
        states = states[:, -model.config.max_length :]
        actions = actions[:, -model.config.max_length :]
        returns_to_go = returns_to_go[:, -model.config.max_length :]
        timesteps = timesteps[:, -model.config.max_length :]

        # pad all tokens to sequence length
        attention_mask = torch.cat(
            [torch.zeros(model.config.max_length - states.shape[1]), torch.ones(states.shape[1])]
        )
        attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
        states = torch.cat(
            [
                torch.zeros(
                    (states.shape[0], model.config.max_length - states.shape[1], model.config.state_dim),
                    device=states.device,
                ),
                states,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        actions = torch.cat(
            [
                torch.zeros(
                    (actions.shape[0], model.config.max_length - actions.shape[1], model.config.act_dim),
                    device=actions.device,
                ),
                actions,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [
                torch.zeros(
                    (returns_to_go.shape[0], model.config.max_length - returns_to_go.shape[1], 1),
                    device=returns_to_go.device,
                ),
                returns_to_go,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        timesteps = torch.cat(
            [
                torch.zeros(
                    (timesteps.shape[0], model.config.max_length - timesteps.shape[1]), device=timesteps.device
                ),
                timesteps,
            ],
            dim=1,
        ).to(dtype=torch.long)
    else:
        attention_mask = None

    _, action_preds, _ = model(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]


def weight_average(model_1, model_2, p, include=None, exclude=None):
    module_1 = model_1
    module_2 = model_2

    module_1.cpu()
    module_2.cpu()

    param_names = module_1.state_dict().keys()
    print("parameter names", param_names)


    # Keep original state_dicts
    params_1 = copy.deepcopy(module_1.state_dict())
    params_2 = copy.deepcopy(module_2.state_dict())

    # Get names of parameters that we will mix (may want to only mix some portions of the network)
    param_names_subset = get_subset(param_names, include=include, exclude=exclude)

    # # Calculate size of subset and original model
    model_size = sum([value.numel() for value in module_1.state_dict().values()])
    subset_size = sum([value.numel() for name, value in module_1.state_dict().items() if name in param_names_subset])
    print("model size", model_size)
    frac = subset_size/model_size
    print(f"subset size {subset_size} fraction {frac}")

    mixed_params = lerp(p, params_1, params_2, subset=param_names_subset)
    module_1.load_state_dict(mixed_params)
    
    return module_1

def weight_average_multi(models, include=None, exclude=None):
    module_1 = models[0]

    for model in models:
        model.cpu()

    param_names = module_1.state_dict().keys()
    print("parameter names", param_names)


    # Keep original state_dicts
    params = [copy.deepcopy(model.state_dict()) for model in models]

    # Get names of parameters that we will mix (may want to only mix some portions of the network)
    param_names_subset = get_subset(param_names, include=include, exclude=exclude)

    # # Calculate size of subset and original model
    model_size = sum([value.numel() for value in module_1.state_dict().values()])
    subset_size = sum([value.numel() for name, value in module_1.state_dict().items() if name in param_names_subset])
    print("model size", model_size)
    frac = subset_size/model_size
    print(f"subset size {subset_size} fraction {frac}")

    mixed_params = lerp_multi(params, subset=param_names_subset)
    module_1.load_state_dict(mixed_params)
    
    return module_1

def get_save_path(model_str, batch_size, num_batches, make_dir=True):
    # Setup save directory
    model_name = os.path.basename(model_str)
    dir_name = os.path.dirname(model_str)
    save_dir = os.path.join(dir_name, 'info')

    if make_dir:
        # check if save_dir exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    suffix = f'-bs{batch_size}-nb{num_batches}'
    save_name = model_name[:model_name.index('.pt')]
    save_name = f'{save_name}{suffix}'
    save_path = os.path.join(save_dir, save_name)

    return save_path


def setup_fisher_many(models, variant, include, exclude=None):
    batch_size = variant['f_batch_size']
    num_batches = variant['f_num_batches']
    model_paths = variant['pretrained_models']
    
    #get infos
    infos = []
    for model_str in model_paths:
        save_path = get_save_path(model_str, batch_size, num_batches, make_dir=False)
        info = torch.load(save_path)
        infos.append(info)

    assert len(infos) == len(models), "num fish info matrices does not match num models"

    model_1 = models[0]

    for model in models:
        model.cpu()

    param_names = model_1.state_dict().keys()
    print("parameter names", param_names)


    # Keep original state_dicts
    state_dicts = [copy.deepcopy(model.state_dict()) for model in models]

    # Get names of parameters that we will mix (may want to only mix some portions of the network)
    param_names_subset = get_subset(param_names, include=include, exclude=exclude)

    transformer_state_dict = fisher_average_multi(state_dicts, infos, subset=param_names_subset)
    model_1.load_state_dict(transformer_state_dict)
    return model_1

def setup_multi(models, env_data, variant):
    final_model = models[0]

    # mix transformers
    transformer_modules = [model.transformer.h for model in models]
    
    # get environment correspondence
    finetuned_envs = []
    for model in models:
        assert len(model.specializations) == 1, "need specialization"
        finetuned_envs.append(model.specializations[0])

    assert len(set(finetuned_envs)) == 3, "expecting 3 unique finetuned envs"
    #import pdb; pdb.set_trace()

    # setup fisher
    if variant['use_fisher']:
        fisher_infos = []
        for model in models:
            fisher_infos.append(compute_fisher_multi(model, env_data, get_batch,  variant['f_num_batches'], variant['f_batch_size'], variant, variant['device']))
            model.cpu()
        state_dicts = [model.state_dict() for model in models]
        # add assert ack, TODO?
        transformer_state_dict = fisher_average_multi(state_dicts, fisher_infos, disable_trace_norm=variant['no_trace_norm'])
        # for key in transformer_state_dict:
        #     assert 'transformer.h' in key, f"{key}, invalid parameter"
        final_model.load_state_dict(transformer_state_dict, strict=False)
    else:
        # check here to make sure this works?
        final_model.transformer.h = weight_average_multi(transformer_modules)

    # setup input/output projections
    for i, model in enumerate(models):
        target_index = final_model.env_ids.index(finetuned_envs[i])
        source_index = model.env_ids.index(finetuned_envs[i])
        final_model.embed_returns[target_index] = model.embed_returns[source_index]
        final_model.embed_states[target_index] = model.embed_states[source_index]
        final_model.embed_actions[target_index] = model.embed_actions[source_index]
        final_model.embed_lns[target_index] = model.embed_lns[source_index]
        final_model.predict_actions[target_index] = model.predict_actions[source_index]
    
    return final_model

def get_optimizer(args, model):
    if args["custom_transformer_decay"]:
        optimizer = optim.AdamW(
            [
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformer" in str(type(module)).lower())
                                    or ("dataparallel" in str(type(module)).lower())
                                )
                            ]
                        )
                    ),
                    "lr": args["lm_learning_rate"]
                    if args["lm_learning_rate"] is not None
                    else args["learning_rate"],
                    "weight_decay": 0.0,
                },
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformer" not in str(type(module)))
                                    and (
                                        "dataparallel" not in str(type(module)).lower()
                                    )
                                )
                            ]
                        )
                    ),
                    "weight_decay": args["weight_decay"],
                },
            ],
            lr=args["learning_rate"],
            eps=1e-6,
        )
    elif args["pretrained_lm"]:
        optimizer = optim.AdamW(
            [
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformers" in str(type(module)).lower())
                                    or ("dataparallel" in str(type(module)).lower())
                                )
                            ]
                        )
                    ),
                    "lr": args["lm_learning_rate"]
                    if args["lm_learning_rate"] is not None
                    else args["learning_rate"],
                    "weight_decay": 0.0,
                },
                {
                    "params": list(
                        chain(
                            *[
                                list(
                                    (
                                        filter(
                                            lambda p: p.requires_grad,
                                            module.parameters(),
                                        )
                                    )
                                )
                                for module in model.children()
                                if (
                                    ("transformers" not in str(type(module)))
                                    and (
                                        "dataparallel" not in str(type(module)).lower()
                                    )
                                )
                            ]
                        )
                    ),
                    "weight_decay": args["weight_decay"],
                },
            ],
            lr=args["learning_rate"],
            eps=1e-6,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
        )
    return optimizer

# def load_model(path, device):
#     model = torch.load(path, map_location=device)
#     return model
def load_model(name, device):
    loaded_model = torch.load(name, map_location=device)
    if hasattr(loaded_model, "transformer_model"):
        loaded_model.transformer = loaded_model.transformer_model.transformer
        del loaded_model.transformer_model
    # share i/o projection
    if loaded_model.predict_action is None:
        loaded_model.predict_state = lambda x: F.linear(x, loaded_model.embed_state.weight.t())
        loaded_model.predict_return = lambda x: F.linear(x, loaded_model.embed_return.weight.t())
        loaded_model.predict_action = lambda x: F.tanh(
            F.linear(x, loaded_model.embed_action.weight.t())
        )
    return loaded_model


def env_from_model_path(path, args):
    path_divided = path.split('/')
    short = path_divided[2]
    model_name = path_divided[3]
    #dataset = model_name.split('-')[-2]
    dataset = model_name.split('-')[3:-1]
    if len(dataset) > 1:
        dataset = '-'.join(dataset)
    else:
        dataset = dataset[0]
    env_name = None
    for env,info in env_config.items():
        if short == info['short']:
            env_name = env
            break
    assert (env_name is not None), f'env with short {short} not found'

    # TODO
    env = gym.make(env_name)
    # load dataset
    data_path = args.data_path
    dataset_path = f"{data_path}/{short}-{dataset}-v2.pkl"
    with open(dataset_path, "rb") as f:
        states = []
        trajectories = pickle.load(f)
        for path in trajectories:
            states.append(path["observations"])
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    env.state_mean = state_mean
    env.state_std = state_std
    env.short = short
    env.dataset_name = dataset

    return env

def get_subset(names, exclude=None, include=None):
    processed_names = []
    if type(exclude) == str:
        exclude = [exclude]
    if type(include) == str:
        include = [include]
    if include is not None and len(include) == 0:
        include = None
    if exclude is not None and len(exclude) == 0:
        exclude = None
    if exclude is not None:
        for name in names: 
            add = True
            for e in exclude:
                if e in name:
                    add = False
                    break
            if add:
                processed_names.append(name)
        return processed_names
    elif include is not None:
        for name in names:
            for i in include:
                if i in name:
                    processed_names.append(name)
                    break
        return processed_names
    else:
        return names


def map_params(params):
    i = 0
    while True:
        test_str = f'transformer.h.{i}.attn.c_attn.weight'
        if test_str not in params:
            break

        c_attn_weight = params[f'transformer.h.{i}.attn.c_attn.weight']
        c_attn_bias = params[f'transformer.h.{i}.attn.c_attn.bias']

        dim = c_attn_bias.shape[0] // 3

        # Split into keys, queries, values
        for j in range(3):
            params[f'transformer.h.{i}.attn.c_attn.{j}.weight'] = c_attn_weight[:, (dim*j):(dim*(j+1))]
            params[f'transformer.h.{i}.attn.c_attn.{j}.bias'] = c_attn_bias[(dim*j):(dim*(j+1))]

        # Delete old
        del params[f'transformer.h.{i}.attn.c_attn.weight']
        del params[f'transformer.h.{i}.attn.c_attn.bias']

        # params[f'transformer.h.{i}.mlp.c_fc.weight'] = params[f'transformer.h.{i}.mlp.c_fc.weight'].T
        # params[f'transformer.h.{i}.mlp.c_proj.weight'] = params[f'transformer.h.{i}.mlp.c_proj.weight'].T

        i += 1

    print(f'mapped transformer with {i} layers')
    return params

def map_subset(param_names_subset, n_blocks):
    param_names_subset = copy.copy(param_names_subset)
    b = []
    for i in range(n_blocks):
        test_str = f'transformer.h.{i}.attn.c_attn.weight'
        if test_str not in param_names_subset:
            continue


        # Split into keys, queries, values
        for j in range(3):
            param_names_subset.append(f'transformer.h.{i}.attn.c_attn.{j}.weight')
            param_names_subset.append(f'transformer.h.{i}.attn.c_attn.{j}.bias')

        # Delete old
        param_names_subset.remove(f'transformer.h.{i}.attn.c_attn.weight')
        param_names_subset.remove(f'transformer.h.{i}.attn.c_attn.bias')

        b.append(i)

    print(f'mapped subset of transformer with layers {b}')
    return param_names_subset

def original_params(params, n_blocks):
    b = []
    for i in range(n_blocks):
        test_str = f'transformer.h.{i}.attn.c_attn.0.weight'
        if test_str not in params:
            continue

        original_bias = torch.concat([params[f'transformer.h.{i}.attn.c_attn.{j}.bias'] for j in range(3)])
        original_weight = torch.concat([params[f'transformer.h.{i}.attn.c_attn.{j}.weight'] for j in range(3)], dim=1)
        
        params[f'transformer.h.{i}.attn.c_attn.weight'] = original_weight
        params[f'transformer.h.{i}.attn.c_attn.bias'] = original_bias

        for j in range(3):
            del params[f'transformer.h.{i}.attn.c_attn.{j}.weight']
            del params[f'transformer.h.{i}.attn.c_attn.{j}.bias']
        

        b.append(i)
    print(f'undid mapping with layers {b}')
    return params

def load_data(data_path, env_name, dataset, variant, settings: dict):
    dataset_path =  f"{data_path}/{env_name}-{dataset}-v2.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    mode = variant.get("mode", "normal")
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            path["rewards"][-1] = path["rewards"].sum()
            path["rewards"][:-1] = 0.0
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    # return dictionary
    data = {
        'state_mean': state_mean,
        'state_std': state_std, 
        'trajectories': trajectories,
        'num_trajectories': len(trajectories),
        'p_sample': traj_lens / sum(traj_lens),
        'state_dim': trajectories[0]['observations'].shape[-1],
        'action_dim': trajectories[0]['actions'].shape[-1],
    }
    for key in settings:
        data[key] = settings[key]
    return data

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def get_batch(data, batch_size, variant, device):
    trajectories = data['trajectories']
    num_trajectories = data['num_trajectories']
    state_mean = data['state_mean']
    state_std = data['state_std']
    state_dim = data['state_dim']
    act_dim = data['action_dim']
    max_ep_len = data['max_ep_len']
    max_len = variant["K"]
    scale = data['scale']
    

    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p=data['p_sample'],  # reweights so we sample according to timesteps
    )

    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        traj = trajectories[batch_inds[i]]
        si = random.randint(0, traj["rewards"].shape[0] - 1)

        # get sequences from dataset
        s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
        a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
        r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
        if "terminals" in traj:
            d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
        else:
            d.append(traj["dones"][si : si + max_len].reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = (
            max_ep_len - 1
        )  # padding cutoff
        rtg.append(
            discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                : s[-1].shape[1] + 1
            ].reshape(1, -1, 1)
        )
        if rtg[-1].shape[1] <= s[-1].shape[1]:
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate(
            [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
        )
        s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate(
            [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
        )
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = (
            np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
            / scale
        )
        timesteps[-1] = np.concatenate(
            [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
        )
        mask.append(
            np.concatenate(
                [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
            )
        )

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(
        dtype=torch.float32, device=device
    )
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(
        dtype=torch.float32, device=device
    )
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(
        dtype=torch.float32, device=device
    )
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(
        dtype=torch.long, device=device
    )
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
        dtype=torch.float32, device=device
    )
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
        dtype=torch.long, device=device
    )
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a, r, d, rtg, timesteps, mask

def lerp_multi(t, subset=None):
  t_out = copy.deepcopy(t[0])
  num_models = len(t)
  assert (num_models > 1), "need more than 1 model"
  prop = 1/num_models
  for p in t[0]: 
    if (subset is None or p in subset):
      #t3[p] = (1 - lam) * t1[p] + lam * t2[p]
      t_out[p] = prop * t[0][p]
      for i in range(1, num_models):
        t_out[p] += prop * t[i][p]

  return t_out

def lerp(lam, t1, t2, subset=None):
    t3 = copy.deepcopy(t1)
    for p in t1: 
        if (subset is None or p in subset):
            t3[p] = (1 - lam) * t1[p] + lam * t2[p]

    return t3

def fisher_average(t1, t2, info_1, info_2, dim_1, dim_2, subset=None):
    weights = np.array([0.,0.])
    t3 = copy.deepcopy(t1)
    for p in t1: 
        if p not in info_1:
            continue
        if (subset is None or p in subset):
            f_1 = info_1[p]
            f_2 = info_2[p]
            f_1 = f_1 / (info_1['_stats']['sum'] )
            f_2 = f_2 / (info_2['_stats']['sum'] )

            f_total = f_1 + f_2
            weights[0] += (f_1 / f_total).mean().item()
            weights[1] += (f_2 / f_total).mean().item()

            t3[p] = (f_1 / f_total) * t1[p] +  (f_2 / f_total) * t2[p]
    print(weights)
    return t3

def fisher_average_multi(t, info, subset=None, disable_trace_norm=False):
    weights = np.zeros(len(t), dtype=np.float32)
    t_out = copy.deepcopy(t[0])
    n_models = len(t)
    assert n_models > 1, "need more than one model to merge"

    for p in t[0]: 
        if p not in info[0]:
            continue
        if (subset is None or p in subset):
            # store FI of param for each task, scale
            #f = [i[p] / i[p].std() for i in info]
            #f = [i[p] for i in info]
            if disable_trace_norm:
                f = [i[p] for i in info]
            else:
                f = [i[p] / i['_stats']['sum'] for i in info]

            f_total = torch.sum(torch.stack(f), dim=0)
            
            for i in range(len(f)):
                weights[i] += (f[i] / f_total).mean().item()
            #t_out[p] = (f_1 / f_total) * t1[p] +  (f_2 / f_total) * t2[p]
            # t_out[p] = (f[0]  / f_total) * t[0][p]
            # for i in range(1, len(f)):
            #     t_out[p] += (f[i] / f_total) * t[i][p]
            t_out[p] = torch.sum(torch.stack([(f[i] / f_total) * t[i][p] for i in range(n_models)]), dim=0)
            #t_out[p] = (t[0][p] + t[1][p] + t[2][p]) / 3
            
    print(weights)
    return t_out



def load_data_compute_fisher(model, env, args):
    variant = {'K': args.K}
    settings = {'max_ep_len': 1000, 'scale': 1000}
    data = load_data(args.data_path, env.short, env.dataset_name, variant, settings)

    # compute fisher
    fisher_info = compute_fisher(model, data, get_batch, args.f_num_batches, args.f_batch_size, variant, args.device)
    return fisher_info
# 
def compute_fisher_multi(model, data, get_batch, n_batches, batch_size, variant, device):
    model.to(device=device)
    fisher_infos = {}
    fisher_info = {}
    assert len(data.keys()) == 3, "must have three envs"
    if variant['spec_info']:
        print("spec info")
        assert len(model.specializations) == 1, "model must have specialization"
        model.toggle_env(model.specializations[0])
        fisher_info = compute_fisher(model, data[model.specializations[0]], get_batch, n_batches, batch_size, variant, device)
    else:
        for env in data.keys():
            model.toggle_env(env)
            f_i = compute_fisher(model, data[env], get_batch, n_batches, batch_size, variant, device)
            fisher_infos[env] = f_i
        # Average FI
        # iterate over three dictionaries simulatenous which share keys
        for (n_1, p_1), (n_2, p_2), (n_3, p_3) in zip(*[fisher_infos[env].items() for env in data.keys()]):
            assert (n_1 == n_2) and (n_2 == n_3), "params must match"
            if type(p_1) == dict:
                continue
            fisher_info[n_1] = (p_1 + p_2 + p_3) / 3
        fisher_info['_stats'] = {}
        count = 0
        for fi in fisher_infos.values():
            count += 1
            fisher_info['_stats']['sum'] = fisher_info['_stats'].get('sum', 0) + fi['_stats']['sum']
            fisher_info['_stats']['mean'] = fisher_info['_stats'].get('mean', 0) + fi['_stats']['mean']
        assert (count == 3), "should have fi dict for each env"
        fisher_info['_stats']['mean'] /= count
    return fisher_info

def compute_fisher(model, data, get_batch, n_batches, batch_size, variant, device):
    fisher_info = {
        n: torch.zeros_like(p).to(device=device) for n, p in model.named_parameters() if p.requires_grad and 'transformer.h' in n
    }

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    n_samples = 0
    print("Computing Fisher...")
    losses = []
    for i in tqdm(range(n_batches)):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = get_batch(data, batch_size, variant, device)

        # Predict actions
        state_preds, action_preds, reward_preds, all_embs = model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )
        action_target = torch.clone(actions)
        
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]
        n_samples += action_target.shape[0]

        # Compute loss
        loss = torch.mean((action_preds - action_target) ** 2)

        #loss = torch.log(torch.exp( (torch.pi ** (-act_dim/2) ) * torch.mean((action_preds - action_target) ** 2)))

        # from torch.distributions import Normal, Independent

        # action_distributions = Independent(Normal(action_preds, torch.ones_like(action_preds) * 1e-3),1)
        # eps = torch.finfo(action_target.dtype).eps
        # action_target = torch.clamp(action_target, -1+eps, 1-eps)
        # action_log_probs = action_distributions.log_prob(action_target)
        # #import pdb; pdb.set_trace()
        # loss = -action_log_probs.mean()
        losses.append(loss.cpu().detach().item())
        optimizer.zero_grad()
        loss.backward()

        # add squared gradients
        for n, p in model.named_parameters():
            if 'transformer.h' not in n:
                continue
            if p.requires_grad and p.grad is not None:
                fisher_info[n] += p.grad.clone().pow(2)

    print(f"Computed fisher over {n_samples} samples")
    ## Average
    total_info = 0
    n_params = 0
    for n,p in fisher_info.items():
        # compute FI
        fisher_info[n] = (p / n_samples).cpu()


        total_info += fisher_info[n].sum()
        n_params += fisher_info[n].numel()
    average_info = total_info / n_params


    print("average FI", average_info)
    fisher_info['_stats'] = {
        'mean': average_info,
        'sum': total_info,
    }
    mean_loss = np.mean(losses)
    print(fisher_info['_stats'])
    print("mean loss", mean_loss)
    print("ratio", mean_loss / average_info)
    print("total info", total_info)
    return fisher_info


def get_module_by_name(module: Union[torch.Tensor, nn.Module],
                       access_string: str):
    """Retrieve a module nested in another by its access string.

    Works even when there is a Sequential in the module.
    credit: https://discuss.pytorch.org/t/how-to-access-to-a-layer-by-module-name/83797/8
    """
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)
