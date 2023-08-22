# MultiDecisionTransformer version of experiment.py

import gym
import numpy as np
import torch
import wandb
import pathlib

import argparse
import pickle
import random
import sys
import copy


from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.multi_decision_transformer import MultiDecisionTransformer
from lm_cotraining.decision_transformer.models.decision_transformer import DecisionTransformer as LmDecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.training.multi_seq_trainer import MultiSequenceTrainer

from transformers import DecisionTransformerModel
from dt_utils import get_action, env_config, weight_average, load_model, weight_average_multi, setup_multi
from dt_utils import load_data, get_batch, discount_cumsum, get_module_by_name
from dt_utils import get_optimizer

import os

def get_dataset(model_name):
    split_name = model_name.split('-')
    dataset = split_name[3:-1]
    if len(dataset) > 1:
        dataset = '-'.join(dataset)
    else:
        dataset = dataset[0]
    return dataset



def experiment(
    exp_prefix,
    variant,
):  
    if variant["seed"] is not None:
        torch.manual_seed(variant["seed"])
    
    device = variant.get("device", "cuda")
    log_to_wandb = variant.get("log_to_wandb", False)

    
    model_type = variant["model_type"]
    assert model_type in ["dt"]

    env_names = ['hopper', 'halfcheetah', 'walker2d']
    env_ids = ['Hopper-v3', 'HalfCheetah-v3', 'Walker2d-v3']
    dataset = variant["dataset"]

    mode = variant.get("mode", "normal")

    group_name = f"{exp_prefix}-multi-{dataset}"
    exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"

    model_dir = os.path.join(pathlib.Path(__file__).parent.resolve(),f'./models/multi/')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Setup environments for each env
    envs = [gym.make(env_id) for env_id in env_ids]
    env_targets = [env_config[env_id]['target_return'] for env_id in env_ids]
    scales = [1000.0] * 3
    max_ep_lens = [1000] * 3


    # Dictionary containg inner-dictionaries for each environmentm which have trajectories, etc
    env_data = {}
    for i in range(len(env_ids)):
        settings = {}
        # set env targets
        settings['target']  = env_targets[i]
        settings['max_ep_len'] = max_ep_lens[i]
        env_data[env_ids[i]] = load_data(variant['data_path'], env_names[i], dataset, variant, settings)
        env_data[env_ids[i]]['scale'] = scales[i]

    if variant['norm_env']:
        old_dataset = variant['norm_env']
        if old_dataset != dataset:
                print("Using old state statistics")
                # update env_data

                # load stats
                state_stats = pickle.load(open(os.path.join(variant['data_path'], 'state_stats.pkl'),'rb'))
                for i in range(len(env_ids)):
                    env_name = env_names[i]
                    env_str = f'{env_name}-{old_dataset}-v2'
                    env_data[env_ids[i]]['state_mean'] = state_stats[env_str]['state_mean']
                    env_data[env_ids[i]]['state_std'] = state_stats[env_str]['state_std']
    else:
        # Check if pre-trained model has different dataset than current finetuning
        old_model_path = variant['pretrained_model'] or variant['load_checkpoint']
        if old_model_path:
            old_dataset = get_dataset(old_model_path)

            if old_dataset != dataset:
                print("Using old state statistics")
                # update env_data

                # load stats
                state_stats = pickle.load(open(os.path.join(variant['data_path'], 'state_stats.pkl'),'rb'))
                for i in range(len(env_ids)):
                    env_name = env_names[i]
                    env_str = f'{env_name}-{old_dataset}-v2'
                    env_data[env_ids[i]]['state_mean'] = state_stats[env_str]['state_mean']
                    env_data[env_ids[i]]['state_std'] = state_stats[env_str]['state_std']
                

    K = variant["K"]

    num_eval_episodes = variant['num_eval_episodes']
    def eval_episodes(target_rew, env_id):
        env_index = env_ids.index(env_id)
        state_dim = env_data[env_id]['state_dim']
        act_dim = env_data[env_id]['action_dim']
        scale = scales[env_index]
        max_ep_len = env_data[env_id]['max_ep_len']
        state_mean = env_data[env_id]['state_mean']
        state_std = env_data[env_id]['state_std']
        def fn(model):
            #assert (eval_env_id == env_id) # maybe change
            #print(env_id)
            model.toggle_env(env_id)
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        envs[env_index],
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew / scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f"target_{target_rew}_{env_id}_return_mean": np.mean(returns),
                f"target_{target_rew}_{env_id}_return_std": np.std(returns),
                f"target_{target_rew}_{env_id}_length_mean": np.mean(lengths),
                f"target_{target_rew}_{env_id}_length_std": np.std(lengths),
            }

        return fn

    # Load or initialize model
    multi_paths = variant['multi_pretrained_models']
    if multi_paths:
        assert len(multi_paths) == 3, "pre-trained model for each env"
        models = [load_model(multi_paths[i], device=device) for i in range(len(multi_paths))]
        model = setup_multi(models, env_data, variant)
        # Merge
    elif variant['pretrained_model']:
        model = torch.load(variant['pretrained_model'], map_location=device)
        assert type(model) == MultiDecisionTransformer
    else:
        model = MultiDecisionTransformer(
            args=variant,
            envs=envs,
            max_length=K,
            max_ep_len=max(max_ep_lens),
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=0.1,
        )
        # or load from state dict
        if variant["load_checkpoint"]:
            state_dict = torch.load(variant["load_checkpoint"])
            model.load_state_dict(state_dict)
            print(f"Loaded from {variant['load_checkpoint']}")
        #print("transformer params", sum([value.numel() for name, value in model.transformer.h.state_dict().items() if value.requires_grad]))
    print("transformer params", sum([param.numel() for param in model.transformer.h.parameters() if param.requires_grad]))
    # Keep pre_trained weights
    model.transformer_pre = copy.deepcopy(model.transformer.h.state_dict())

    for env_id in env_ids:
        model.toggle_env(env_id)
        model = model.to(device=device)
    
    # optionally freeze subset, useful for merging finetuned
    total_params = 0
    frozen_params = 0
    if variant['freeze_subset']:
        # iterate over parameters
        for name, param in model.named_parameters():
            disable_grad = False
            for i in variant['freeze_subset']:
                if i in name:
                    disable_grad = True
                    break
            if disable_grad:
                param.requires_grad = False
                frozen_params += param.numel()
            total_params += param.numel()
    #print(frozen_params/total_params, "prop frozen")

    # Setup optimizer
    warmup_steps = variant["warmup_steps"]
    optimizer = get_optimizer(args=variant, model=model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    # Check if only training on subset of envs
    if variant['selected_training']:
        training_ids = []
        for env_name in variant['selected_training']:
            assert env_name in env_names, "invalid env name passed"
            env_id = env_ids[env_names.index(env_name)]
            training_ids.append(env_id)
            if not hasattr(model, 'specializations'):
                model.specializations = []
            model.specializations.append(env_id)
    else:
        training_ids = env_ids

    if not variant['eval_only']:
        assert len(training_ids) > 0, "need at least one training environment"
    
    print("training ids", training_ids)

    # TODO, make MultiSequenceTrainer
    trainer = MultiSequenceTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            env_data=env_data,
            batch_size=variant["batch_size"],
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(target, env_id) for target, env_id in zip(env_targets, env_ids)],
            training_ids=training_ids,
            eval_only=variant["eval_only"],
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project=variant['project'],
            config=variant,
        )
    # TODO, change metri
    #base_metrics = [f'evaluation/target_{target}_return_mean' for target in env_targets]
    metric = f'evaluation/average_norm_return_mean'
    model.training_checkpoint_dict = {
        'best_state_dict': None,
        'best_metric': None
    }
    # TODO add training loop, saving

    for iter in range(variant["max_iters"]):
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"], iter_num=iter + 1, print_logs=True
        )
        if variant['save_mode'] == 'best' or variant['save_mode'] == 'both':
            metric_value = outputs[metric]
            if model.training_checkpoint_dict['best_metric'] is None or metric_value > model.training_checkpoint_dict['best_metric']:
                model.training_checkpoint_dict['best_metric'] = metric_value
                if 'best_state_dict' in outputs:
                    model.training_checkpoint_dict['best_state_dict_h'] = outputs['best_state_dict']
                    model.training_checkpoint_dict['best_state_dict'] = copy.deepcopy(model.state_dict())
                else:
                    model.training_checkpoint_dict['best_state_dict'] = copy.deepcopy(model.state_dict())
                print("Updated best model")
        if 'best_state_dict' in outputs:
            del outputs['best_state_dict']
        if log_to_wandb:
            wandb.log(outputs)

    training_checkpoint_dict = model.training_checkpoint_dict
    model.training_checkpoint_dict = {}

    if variant['save_model']:
        if variant['save_mode'] == 'both' or variant['save_mode'] == 'last':
            model_name = model_type + '_' + exp_prefix + '.pt'
            print("saving", model_name)
            torch.save(model,os.path.join(model_dir, model_name))
        
        if variant['save_mode'] == 'both' or variant['save_mode'] == 'best':
            best_metric = training_checkpoint_dict['best_metric']
            print(f'Saving best model with metric {best_metric}')
            model.load_state_dict(training_checkpoint_dict['best_state_dict'])
            if 'best_state_dict_h' in training_checkpoint_dict:
                print("overwriting transformer.h")
                model.transformer.h.load_state_dict(training_checkpoint_dict['best_state_dict_h'])
            torch.save(model,os.path.join(model_dir, model_type + '_' + exp_prefix + '_'  +'best' + '.pt'))

    if variant['end_break']:
        import pdb; pdb.set_trace()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--project', type=str, default="multi-decision-transformer")
    parser.add_argument('--note', default=None, type=str)
    #parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument(
        "--dataset", type=str, default="medium"
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--model_type", type=str, default="dt"
    )  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument('--disable_attention', default=False, action='store_true')

    parser.add_argument('--use_fisher', default=False, action='store_true')
    parser.add_argument('--spec_info', default=False, action='store_true')
    parser.add_argument('--no_trace_norm', default=False, action='store_true')
    parser.add_argument('--f_batch_size', default=64, type=int)
    parser.add_argument('--f_num_batches', default=100, type=int)

    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--lm_learning_rate", "-lmlr", type=float, default=None)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument('--init_reg', default=None, type=float)
    parser.add_argument('--custom_transformer_decay', default=False, action='store_true')

    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=40)
    parser.add_argument("--num_steps_per_iter", type=int, default=2500)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wandb", "-w", action="store_true", default=False)

    parser.add_argument('--selected_training', default=None, nargs='+', type=str)
    parser.add_argument('--score_selected', default=False, action='store_true')

    parser.add_argument('--freeze_subset', default=None, nargs='+', type=str)

    parser.add_argument('--eval_merge_pre', default=False, action='store_true')
    parser.add_argument('--p_list', nargs='+', type=float, default=[0.5])

    #parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument('--end_break', default=False, action='store_true')
    parser.add_argument("--share_input_output_proj", action="store_true", default=False)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--frozen", action="store_true", default=False)
    parser.add_argument("--extend_positions", action="store_true", default=False)
    parser.add_argument("--pretrained_lm", type=str, default=None)
    parser.add_argument('--pretrained_model', default=None, type=str)
    parser.add_argument('--multi_pretrained_models', default=None, type=str, nargs='+')
    parser.add_argument('--norm_env', default=None, choices=['medium', 'medium-expert', 'expert', 'medium-replay'])
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_mode', default='last', choices=['last', 'best', 'both'])

    args = parser.parse_args()

    experiment("gym-experiment", variant=vars(args))
