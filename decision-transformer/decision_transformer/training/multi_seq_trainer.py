import time
import tqdm
import copy

import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer import Trainer
from dt_utils import lerp

class MultiSequenceTrainer(Trainer):
    def __init__(
        self,
        args,
        model,
        optimizer,
        batch_size,
        get_batch,
        loss_fn,
        env_data,
        training_ids,
        scheduler=None,
        eval_fns=None,
        eval_only=False,
    ):
        super().__init__(
            args,
            model,
            optimizer,
            batch_size,
            get_batch,
            loss_fn,
            scheduler=scheduler,
            eval_fns=eval_fns,
            eval_only=eval_only,
        )
        self.env_data = env_data
        self.env_ids = list(self.env_data.keys())
        self.training_ids = training_ids

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        if not self.eval_only:
            self.model.train()
            for _ in tqdm.tqdm(range(num_steps), desc="Training"):
                train_loss = self.train_step()
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

                logs["time/training"] = time.time() - train_start
                logs["training/train_loss_mean"] = np.mean(train_losses)
                logs["training/train_loss_std"] = np.std(train_losses)

        eval_start = time.time()
        best_state_dict = None
        best_metric = None

        if self.args['eval_merge_pre']:
            # Will change state_dict back to original after evaluation
            self.model.transformer_post = copy.deepcopy(self.model.transformer.h.state_dict())
            assert self.model.transformer_pre is not None, "pre state_dict not set"
            #import pdb; pdb.set_trace()
            for p in self.args['p_list']:
                # merge using p, probability of post transformer weights
                merged_transformer = lerp(p, self.model.transformer_pre, self.model.transformer_post)
                self.model.transformer.h.load_state_dict(merged_transformer)
                self.model.eval()
                for eval_fn in tqdm.tqdm(self.eval_fns, desc="Evaluating"):
                    outputs = eval_fn(self.model)
                    for k, v in outputs.items():
                        logs[f"evaluation/{p}/{k}"] = v
                
                # Compute average normalized return
                scores = []
                env_targets = [self.env_data[env_id]["target"] for env_id in self.env_ids]
                base_metrics = [f'evaluation/{p}/target_{target}_{env_id}_return_mean' for target,env_id in zip(env_targets, self.env_ids)]
                for i in range(len(base_metrics)):
                    base_metric = base_metrics[i]
                    assert base_metric in logs, "metric output mismatch"
                    raw_value = logs[base_metric]
                    score = raw_value / env_targets[i]
                    scores.append(score)
                metric_value = np.mean(scores)
                logs[f"evaluation/{p}/average_norm_return_mean"] = metric_value
                if best_metric is None or metric_value > best_metric:
                    best_metric = metric_value
                    best_state_dict = copy.deepcopy(merged_transformer)


            # restore state_dict
            #import pdb; pdb.set_trace()
            self.model.transformer.h.load_state_dict(self.model.transformer_post)
            logs[f"evaluation/average_norm_return_mean"] = best_metric
        else:
            # Standard evaluation
            self.model.eval()
            for eval_fn in tqdm.tqdm(self.eval_fns, desc="Evaluating"):
                outputs = eval_fn(self.model)
                for k, v in outputs.items():
                    logs[f"evaluation/{k}"] = v

            # Compute average normalized return
            scores = []
            if self.args['score_selected']:
                env_targets = [self.env_data[env_id]["target"] for env_id in self.training_ids]
                base_metrics = [f'evaluation/target_{target}_{env_id}_return_mean' for target,env_id in zip(env_targets, self.training_ids)]
            else:
                env_targets = [self.env_data[env_id]["target"] for env_id in self.env_ids]
                base_metrics = [f'evaluation/target_{target}_{env_id}_return_mean' for target,env_id in zip(env_targets, self.env_ids)]

            for i in range(len(base_metrics)):
                base_metric = base_metrics[i]
                assert base_metric in logs, "metric output mismatch"
                raw_value = logs[base_metric]
                score = raw_value / env_targets[i]
                scores.append(score)
            metric_value = np.mean(scores)
            logs[f"evaluation/average_norm_return_mean"] = metric_value

        if not self.eval_only:
            logs["time/total"] = time.time() - self.start_time
        logs["time/evaluation"] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        if not self.eval_only:
            if self.args.get("outdir"):
                torch.save(
                    self.model.state_dict(),
                    f"{self.args['outdir']}/model_{iter_num}.pt",
                )
        if best_state_dict is not None:
            logs['best_state_dict'] = best_state_dict

        return logs



    def train_step(self):
        total_loss = 0
        for env_id in self.env_ids:

            # Check if env is in training set
            if env_id not in self.training_ids:
                continue

            (
                states,
                actions,
                rewards,
                dones,
                rtg,
                timesteps,
                attention_mask,
            ) = self.get_batch(self.env_data[env_id], self.batch_size, self.args, self.args['device'])
            action_target = torch.clone(actions)

            self.model.toggle_env(env_id)
            state_preds, action_preds, reward_preds, all_embs = self.model.forward(
                states,
                actions,
                rewards,
                rtg[:, :-1],
                timesteps,
                attention_mask=attention_mask,
            )

            self.step += 1
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[
                attention_mask.reshape(-1) > 0
            ]

            loss = self.loss_fn(
                None,
                action_preds,
                None,
                None,
                action_target,
                None,
            )
            total_loss += loss
        
        # loss for moving away from transformer.h
        if self.args['init_reg']:
            #state_dict = self.model.transformer.h.state_dict()
            #init_reg_loss = torch.tensor([ torch.linalg.norm(state_dict[name] - param) for name,param in self.model.transformer_pre.items() if param.dtype != torch.uint8]).sum()
            init_reg_loss = torch.tensor(0.).requires_grad_(True)
            for name, param in self.model.transformer.h.named_parameters():
                init_reg_loss = init_reg_loss + torch.linalg.norm(self.model.transformer_pre[name] - param)
            init_reg_loss = self.args['init_reg'] * init_reg_loss
            self.diagnostics['training/init_reg_loss'] = init_reg_loss.detach().cpu().item()
            total_loss += init_reg_loss


        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return total_loss.detach().cpu().item()
