import os
import sys
sys.path.insert(0, "./")

import torch
import torch.nn as nn
import torch.nn.functional as F
from rlf.algos.il.base_irl import BaseIRLAlgo
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler
import rlf.rl.utils as rutils
import rlf.algos.utils as autils
from collections import defaultdict
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from rlf.args import str2bool
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from rlf.exp_mgr.viz_utils import append_text_to_image
import math
from drail.ddpm import MLPConditionDiffusion

from drail.drail import Discriminator, DRAILDiscrim

class DiffATPDiscriminator(Discriminator):
    def __init__(self, state_dim, action_dim, next_state_dim, args, base_net, num_units=128):
        super().__init__(state_dim, action_dim, args, base_net, num_units)




class DiffATPDiscrim(DRAILDiscrim):
    # def _create_discrim(self):
    #     ob_shape = rutils.get_obs_shape(self.policy.obs_space)
    #     ac_dim = rutils.get_ac_dim(self.action_space)
    #     base_net = self.policy.get_base_net_fn(ob_shape)
    #     #* Change to Diffusion Model
    #     discrim = self.get_discrim(base_net.output_shape[0], ac_dim, self.args, base_net, num_units=self.args.discrim_num_unit)
    #     return discrim.to(self.args.device)

    # def init(self, policy, args):
    #     super().init(policy, args)
    #     self.action_space = self.policy.action_space
    #     self.label_dim = self.args.label_dim

    #     self.discrim_net = self._create_discrim()

    #     self.returns = None
    #     self.ret_rms = RunningMeanStd(shape=())

    #     self.opt = optim.Adam(
    #         self.discrim_net.parameters(), lr=self.args.disc_lr)
    #     if self.args.lr_schedule:
    #         self.scheduler = LambdaLR(self.opt, lr_lambda=self.lr_lambda)

    # # Define a function to calculate the learning rate at each epoch
    # def lr_lambda(self, epoch):
    #     lr_start = 0.00005
    #     lr_end = self.args.disc_lr
    #     x = epoch / self.args.n_drail_epochs 
    #     if x < 0.5:
    #         lr = lr_start + (lr_end - lr_start) * math.sqrt(x)
    #     else:
    #         lr = self.args.disc_lr
    #     return lr

    # def adjust_lr(self, lr):
    #     for param_group in self.opt.param_groups:
    #         param_group["lr"] = lr
    
    # def _get_sampler(self, storage):
    #     agent_experience = storage.get_generator(None,
    #                                              mini_batch_size=self.expert_train_loader.batch_size)
    #     return self.expert_train_loader, agent_experience

    # def _trans_batches(self, expert_batch, agent_batch):
    #     return expert_batch, agent_batch

    # def get_env_settings(self, args):
    #     settings = super().get_env_settings(args)
    #     if not args.drail_state_norm:
    #         settings.ret_raw_obs = True
    #     settings.mod_render_frames_fn = self.mod_render_frames
    #     return settings

    def mod_render_frames(self, frame, env_cur_obs, env_cur_action, env_cur_reward,
            env_next_obs, **kwargs):
        use_cur_obs = rutils.get_def_obs(env_cur_obs)
        use_cur_obs = torch.FloatTensor(use_cur_obs).unsqueeze(0).to(self.args.device)

        # ADDED 250313
        use_n_obs = rutils.get_def_obs(env_next_obs)
        use_n_obs = torch.FloatTensor(use_n_obs).unsqueeze(0).to(self.args.device)

        if env_cur_action is not None:
            use_action = torch.FloatTensor(env_cur_action).unsqueeze(0).to(self.args.device)
            # disc_val = self._compute_disc_val(use_cur_obs, use_action).item()
            # ADDED 250313
            disc_val = self._compute_disc_val(use_cur_obs, use_action, use_n_obs).item()
        else:
            disc_val = 0.0

        frame = append_text_to_image(frame, [
            "Discrim: %.3f" % disc_val,
            "Reward: %.3f" % (env_cur_reward if env_cur_reward is not None else 0.0)
            ])
        return frame

    # def _norm_expert_state(self, state, obsfilt):
    #     if not self.args.drail_state_norm:
    #         return state
    #     state = state.cpu().numpy()

    #     if obsfilt is not None:
    #         state = obsfilt(state, update=False)
    #     state = torch.tensor(state).to(self.args.device)
    #     return state

    # def _trans_agent_state(self, state, other_state=None):
    #     if not self.args.drail_state_norm:
    #         if other_state is None:
    #             return state['raw_obs']
    #         return other_state['raw_obs']
    #     return rutils.get_def_obs(state)

    def _compute_discrim_loss(self, agent_batch, expert_batch, obsfilt):
        expert_actions = expert_batch['actions'].to(self.args.device)
        expert_actions = self._adjust_action(expert_actions)
        expert_states = self._norm_expert_state(expert_batch['state'], obsfilt)
        # ADDED next_state 추가 
        expert_n_states = self._norm_expert_state(expert_batch['next_state'], obsfilt)  

        agent_states = self._trans_agent_state(agent_batch['state'],
                agent_batch['other_state'] if 'other_state' in agent_batch else None)
        agent_actions = agent_batch['action']

        # CHECK: agent_batch 형태 확인
        # print("agent_batch_keys :", agent_batch.keys())
        agent_n_states = self._trans_agent_state(agent_batch['next_state'],
                agent_batch['other_next_state'] if 'other_next_state' in agent_batch else None)
        # print("agent_n_states :", agent_n_states)

        agent_actions = rutils.get_ac_repr(
            self.action_space, agent_actions)
        expert_actions = rutils.get_ac_repr(
            self.action_space, expert_actions)

        # ADDED
        # expert_d = self._compute_disc_val(expert_states, expert_actions)
        # agent_d = self._compute_disc_val(agent_states, agent_actions)
        # grad_pen = self.compute_pen(expert_states, expert_actions, agent_states,
        #         agent_actions)

        expert_d = self._compute_disc_val(expert_states, expert_actions, expert_n_states)
        agent_d = self._compute_disc_val(agent_states, agent_actions, agent_n_states)
        grad_pen = self.compute_pen(expert_states, expert_actions, expert_n_states, 
                                    agent_states, agent_actions, agent_n_states)

        return expert_d, agent_d, grad_pen

    def compute_pen(self, expert_states, expert_actions, expert_n_states, agent_states, agent_actions, agent_n_states):
        if self.args.disc_grad_pen != 0.0:
            # CHECK: wass_grad_pen 함수 재정의 필요
            # ADDED
            # grad_pen = self.args.disc_grad_pen * autils.wass_grad_pen(expert_states,
            #         expert_actions, agent_states, agent_actions,
            #         self.args.action_input, self._compute_disc_val)
            grad_pen = self.args.disc_grad_pen * autils.wass_grad_pen(expert_states,
                    expert_actions, expert_n_states, agent_states, agent_actions, agent_n_states
                    self.args.action_input, self._compute_disc_val)
            return grad_pen
        return 0

    def _compute_disc_val(self, state, action, n_state, label=None):
        # ADDED
        label_one = self.discrim_net(state, action, n_state, 1.)
        label_zero = self.discrim_net(state, action, n_state, 0.)
        output = F.softmax(torch.stack([-label_one, -label_zero]),dim=0)[0]
        return output
    
    def plot_reward_map(self, i):
        pass
        # x = torch.linspace(0, 10, 100)
        # y = torch.linspace(-2, 2, 100)
        # X, Y = torch.meshgrid(x, y, indexing="ij")
        # X = X.reshape(-1,1).to(self.args.device)
        # Y = Y.reshape(-1,1).to(self.args.device)
        # with torch.no_grad():
        #     s = self._compute_disc_val(X, Y)
        #     eps = 1e-20
        #     if self.args.reward_type == 'airl':
        #         reward = (s + eps).log() - (1 - s + eps).log()
        #     elif self.args.reward_type == 'gail':
        #         reward = (s + eps).log()
        #     elif self.args.reward_type == 'raw':
        #         reward = s
        #     elif self.args.reward_type == 'airl-positive':
        #         reward = (s + eps).log() - (1 - s + eps).log() + 20
        #     elif self.args.reward_type == 'revise':
        #         d_x = (s + eps).log()
        #         reward = d_x + (-1 - (-d_x).log())
        #     else:
        #         raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
        #     reward = reward.view(100, 100).cpu().numpy().T

        # plt.figure(figsize=(8, 5))
        # plt.imshow(reward, extent=[0, 10, -2, 2], cmap="jet", origin="lower", aspect="auto")
        # plt.colorbar()
        # file_path = "./data/imgs/" + self.args.prefix + "_reward_map.png"
        # plt.savefig(file_path)
        # return file_path
    
    def plot_disc_val_map(self, i):
        pass
        # x = torch.linspace(0, 10, 100)
        # y = torch.linspace(-2, 2, 100)
        # X, Y = torch.meshgrid(x, y, indexing="ij")
        # X = X.reshape(-1, 1).to(self.args.device)
        # Y = Y.reshape(-1,1).to(self.args.device)
        # with torch.no_grad():
        #     rewards = []
        #     for _ in range(10):
        #         reward = self._compute_disc_val(X, Y).view(100, 100).cpu().numpy().T
        #         rewards.append(reward)
        #     reward = torch.tensor(rewards).mean(dim=0)
        # plt.figure(figsize=(8, 5))
        # plt.imshow(reward, extent=[0, 10, -2, 2], cmap="jet", origin="lower", aspect="auto")
        # plt.colorbar()
        # file_path = "./data/imgs/" + self.args.prefix + "_disc_val_map.png"
        # plt.savefig(file_path)
        # return file_path

    # def _compute_expert_loss(self, expert_d, expert_batch):
    #     return  F.binary_cross_entropy(expert_d,
    #             torch.ones(expert_d.shape).to(self.args.device))

    # def _compute_agent_loss(self, agent_d, agent_batch):
    #     return  F.binary_cross_entropy(agent_d,
    #             torch.zeros(agent_d.shape).to(self.args.device))

    # def _update_reward_func(self, storage, gradient_clip=False, t=1):
    #     self.discrim_net.train()

    #     log_vals = defaultdict(lambda: 0)
    #     obsfilt = self.get_env_ob_filt()

    #     expert_sampler, agent_sampler = self._get_sampler(storage)
    #     if agent_sampler is None:
    #         # algo requested not to update this step
    #         return {}
    #     n = 0
    #     for epoch_i in range(self.args.n_drail_epochs):
    #         for expert_batch, agent_batch in zip(expert_sampler, agent_sampler):
    #             expert_batch, agent_batch = self._trans_batches(
    #                 expert_batch, agent_batch)
    #             n += 1
    #             expert_d, agent_d, grad_pen = self._compute_discrim_loss(agent_batch, expert_batch, obsfilt)
    #             expert_loss = self._compute_expert_loss(expert_d, expert_batch)
    #             agent_loss = self._compute_agent_loss(agent_d, agent_batch)
                
    #             discrim_loss = expert_loss + agent_loss

    #             if self.args.disc_grad_pen != 0.0:
    #                 if t <= self.args.disc_grad_pen_period:
    #                     log_vals['grad_pen'] += grad_pen.item()
    #                     total_loss = discrim_loss + self.args.disc_grad_pen * grad_pen
    #                 else: 
    #                     log_vals['grad_pen'] += 0
    #                     total_loss = discrim_loss
    #             else:
    #                 total_loss = discrim_loss
                
    #             self.opt.zero_grad()
    #             total_loss.backward()
    #             if gradient_clip:
    #                 torch.nn.utils.clip_grad_norm_(self.discrim_net.parameters(), max_norm=1.0)
    #             self.opt.step()

    #             log_vals['discrim_loss'] += discrim_loss.item()
    #             log_vals['expert_loss'] += expert_loss.item()
    #             log_vals['agent_loss'] += agent_loss.item()
    #             log_vals['expert_disc_val'] += expert_d.mean().item()
    #             log_vals['agent_disc_val'] += agent_d.mean().item()
    #             log_vals['agent_reward'] += ((agent_d + 1e-20).log() - (1 - agent_d + 1e-20).log()).mean().item()
    #             log_vals['dm_update_data'] += len(expert_batch)
    #             self.step += self.expert_train_loader.batch_size
    #     for k in log_vals:
    #         if k[0] != '_':
    #             log_vals[k] /= n
    #     if self.args.env_name[:4] == "Sine" and (self.step // (self.expert_train_loader.batch_size * n)) % 100 == 1 :
    #         # log_vals["_reward_map"] = self.plot_reward_map(self.step)
    #         log_vals["_disc_val_map"] = self.plot_disc_val_map(self.step)

    #     log_vals['dm_update_data'] *= n
    #     return log_vals
    
    def _compute_discrim_reward(self, storage, step, add_info):
        state = self._trans_agent_state(storage.get_obs(step))
        action = storage.actions[step]
        action = rutils.get_ac_repr(self.action_space, action)
        # ADDED
        # CHECK
        ########
        print(type(storage))
        n_state = self._trans_agent_state(storage.get_obs(step+1)) # TO BE MODIFIED
        # step+1이 범위 넘어가는 경우 발생.
        # transition_storage, rollout_storage에 따라 형식 바뀌어야함.
        ########
        
        # ADDED
        # s = self._compute_disc_val(state, action)
        s = self._compute_disc_val(state, action, n_state)
        
        eps = 1e-20
        if self.args.reward_type == 'airl':
            reward = (s + eps).log() - (1 - s + eps).log()
        elif self.args.reward_type == 'gail':
            reward = (s + eps).log()
        elif self.args.reward_type == 'raw':
            reward = s
        elif self.args.reward_type == 'airl-positive':
            reward = (s + eps).log() - (1 - s + eps).log() + 20
        elif self.args.reward_type == 'revise':
            d_x = (s + eps).log()
            reward = d_x + (-1 - (-d_x).log())
        else:
            raise ValueError(f"Unrecognized reward type {self.args.reward_type}")
        return reward

    # def _get_reward(self, step, storage, add_info):
    #     masks = storage.masks[step]
    #     with torch.no_grad():
    #         self.discrim_net.eval()
    #         reward = self._compute_discrim_reward(storage, step, add_info)

    #         if self.args.drail_reward_norm:
    #             if self.returns is None:
    #                 self.returns = reward.clone()

    #             self.returns = self.returns * masks * self.args.gamma + reward
    #             self.ret_rms.update(self.returns.cpu().numpy())

    #             return reward / np.sqrt(self.ret_rms.var[0] + 1e-8), {}
    #         else:
    #             return reward, {}
    
    def eval_disc(self, state, action, n_state, label):
        self.discrim_net.eval()
        with torch.no_grad():
            # ADDED
            # discrim_output = self.discrim_net(state, action, label)
            discrim_output = self.discrim_net(state, action, n_state, label)
        return discrim_output

    # def get_add_args(self, parser):
    #     super().get_add_args(parser)
    #     #########################################
    #     # Overrides

    #     #########################################
    #     # New args
    #     #! TODO: Modify to fit Drail
    #     parser.add_argument('--action-input', type=str2bool, default=False)
    #     parser.add_argument('--drail-reward-norm', type=str2bool, default=False)
    #     parser.add_argument('--drail-state-norm', type=str2bool, default=True)
    #     parser.add_argument('--drail-action-norm', type=str2bool, default=False)
    #     parser.add_argument('--disc-lr', type=float, default=0.0001)
    #     parser.add_argument('--disc-grad-pen', type=float, default=0.0)
    #     parser.add_argument('--disc-grad-pen-period', type=float, default=1.0)
    #     parser.add_argument('--expert-loss-rate', type=float, default=1.0)
    #     parser.add_argument('--agent-loss-rate', type=float, default=-1.0)
    #     parser.add_argument('--agent-loss-rate-scheduler', type=str2bool, default=False)
    #     parser.add_argument('--agent-loss-end', type=float, default=-1.1)
    #     parser.add_argument('--discrim-depth', type=int, default=4)
    #     parser.add_argument('--discrim-num-unit', type=int, default=128)
    #     parser.add_argument('--sample-strategy', type=str, default="random")
    #     parser.add_argument('--sample-strategy-value', type=int, default=250)
    #     parser.add_argument('--n-drail-epochs', type=int, default=1)
    #     parser.add_argument('--label-dim', type=int, default=10)
    #     parser.add_argument('--test-sine-env', type=str2bool, default=False)
    #     parser.add_argument('--deeper-ddpm', type=str2bool, default=False)
    #     parser.add_argument('--reward-type', type=str, default='airl', help="""
    #             One of [Drail]. Changes the reward computation. Does
    #             not change training.
    #             """)

    # def load_resume(self, checkpointer):
    #     super().load_resume(checkpointer)
    #     self.opt.load_state_dict(checkpointer.get_key('drail_disc_opt'))
    #     self.discrim_net.load_state_dict(checkpointer.get_key('drail_disc'))

    # def save(self, checkpointer):
    #     super().save(checkpointer)
    #     checkpointer.save_key('drail_disc_opt', self.opt.state_dict())
    #     checkpointer.save_key('drail_disc', self.discrim_net.state_dict())