import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer import Trainer
import torch_semiring_einsum


EQUATION = torch_semiring_einsum.compile_equation("iaj,bj->iabj")


def kmeans_cosine_max_loss(centers, seq, mean=False):
    assert centers.device == seq.device
    # loss = -(torch.einsum("iaj,bj->iabj", [seq, centers]).max(2).values.mean())
    if mean:
        loss = -(
            torch_semiring_einsum.einsum(EQUATION, seq, centers, block_size=1).mean()
        )
    else:
        loss = -(
            torch_semiring_einsum.einsum(EQUATION, seq, centers, block_size=1)
            .max(2)
            .values.mean()
        )

    print(loss.item())

    return loss


kmeans_anneal = lambda x: 1 / (1 + np.exp(-(((15000 - x) / (15000 / 10)) - 5)))

def lm_linear_anneal(constant, total_steps, current_step):
    return constant * (1 - current_step / total_steps)

class SequenceTrainer(Trainer):
    def train_step(self):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = self.get_batch(self.batch_size)

        if self.args["fp16"]:
            with torch.cuda.amp.autocast():
                if self.args["joint"]:
                    batch = next(self.train_nlp_dataset)
                    lm_out = self.model.transformer_model(**batch)
                    lm_loss = lm_out.loss
                    print(lm_loss)

                action_target = torch.clone(actions)

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
                action_preds = action_preds.reshape(-1, act_dim)[
                    attention_mask.reshape(-1) > 0
                ]
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
                print(loss.item())
                if self.args["joint"]:
                    loss += self.args['gpt_lm_const'] * lm_loss
                if self.args["gpt_kmeans"]:
                    loss += (
                        self.args["gpt_kmeans_const"]
                        * kmeans_anneal(self.step)
                        * kmeans_cosine_max_loss(
                            self.model.cluster_centers,
                            all_embs,
                            mean=self.args["kmeans_mean"],
                        )
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        else:
            if self.args["joint"]:
                # Fixing last batch, wrong size
                try:
                    batch = next(self.train_nlp_dataset)
                except:
                    batch = next(self.train_nlp_dataset)
                batch['input_ids'] = batch['input_ids'].to(device=self.model.transformer.device)
                batch['attention_mask'] = batch['attention_mask'].to(device=self.model.transformer.device)
                batch['labels'] = batch['labels'].to(device=self.model.transformer.device)
                lm_out = self.model.transformer_model(**batch)
                lm_loss = lm_out.loss
                print(lm_loss)
                self.diagnostics["training/lm_loss"] = lm_loss.detach().cpu().item()

            action_target = torch.clone(actions)

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
            action_preds = action_preds.reshape(-1, act_dim)[
                attention_mask.reshape(-1) > 0
            ]
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
            print(loss.item())
            if self.args["joint"]:
                if self.args["lm_const_linear_decay"]:
                    total_steps = self.args['max_iters'] * self.args['num_steps_per_iter']
                    lm_const = lm_linear_anneal(self.args['gpt_lm_const'], total_steps, self.step - 1)
                else:
                    lm_const = self.args['gpt_lm_const']
                self.diagnostics['training/lm_const'] = lm_const
                loss += lm_loss * lm_const
            if self.args["gpt_kmeans"]:
                if self.args['disable_k_annealing']:
                    annealing_value = 1
                else:
                    annealing_value = kmeans_anneal(self.step)

                loss += (
                    self.args["gpt_kmeans_const"]
                    * annealing_value
                    * kmeans_cosine_max_loss(
                        self.model.cluster_centers,
                        all_embs,
                        mean=self.args["kmeans_mean"],
                    )
                )
            if self.init_reg:
                # for name,param in self.init_state.items():
                #     if param.dtype == torch.uint8:
                #         print(name)
                    # if 'transformer' in name:
                    #     try:
                    #         torch.linalg.norm(state_dict[name] - param)
                    #     except:
                    #         import pdb; pdb.set_trace()
                # init_reg_loss = self.init_reg * torch.tensor([ torch.linalg.norm(state_dict[name] - param) for name,param in self.init_state.items() if ('transformer' in name and param.dtype != torch.uint8)]).sum()
                # self.diagnostics['training/init_reg_loss'] = init_reg_loss.detach().cpu().item()
                init_reg_loss = torch.tensor(0.).requires_grad_(True)
                for name, param in self.model.transformer.h.named_parameters():
                    init_reg_loss = init_reg_loss + torch.linalg.norm(self.init_state[name] - param)
                init_reg_loss = self.args['init_reg'] * init_reg_loss
                self.diagnostics['training/init_reg_loss'] = init_reg_loss.detach().cpu().item()

                loss += init_reg_loss

                #import pdb; pdb.set_trace()



            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item()
