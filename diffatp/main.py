import sys
import torch
from collections import defaultdict
sys.path.insert(0, "./")

from rlf import run_policy

#### Newly Addeded ####
from drail.main import DrailSettings, get_setup_dict
import diffatp.atp_envs

import rlf
import rlf.rl.utils as rutils
from rlf.rl.checkpointer import Checkpointer
from rlf.rl.envs import make_vec_envs
import gym
from gym.spaces import Box
from diffatp.diffatp_ppo import DiffATPPPO

from functools import partial
from rlf import run_policy
from rlf.algos import BaseAlgo
from rlf.algos.il.base_il import BaseILAlgo
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.il.sqil import SQIL
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.rl.loggers.wb_logger import WbLogger, get_wb_ray_config, get_wb_ray_kwargs
from rlf.args import str2bool
from rlf.run_settings import RunSettings
from goal_prox.method.utils import trim_episodes_trans
from goal_prox.envs.goal_traj_saver import GoalTrajSaver
from rlf.algos.on_policy.ppo import PPO
from drail.get_policy import get_ppo_policy

from diffatp.custom_drail import DiffATP
from rlf.policies.base_policy import get_step_info

def get_setup_dict():
    return {
        "diffATP": (DiffATP(), get_ppo_policy),  
        "ppo": (PPO(), get_ppo_policy),
    }

class DiffATPRunner(rlf.Runner):
    def training_iter(self, update_iter: int, beginning=False, t=1):
        """
        기본 Runner를 오버라이드하여 storage.insert() 시 24차원 → 23차원 변환
        """
        self.log.start_interval_log()
        self.updater.pre_update(update_iter)
        
        for step in self.updater.get_steps_generator(update_iter):
            # Sample actions
            obs = self.storage.get_obs(step)

            step_info = get_step_info(update_iter, step, self.episode_count, self.args)
            
            with self.train_ctx():
                ac_info = self.policy.get_action(
                    rutils.get_def_obs(obs, self.args.policy_ob_key),
                    rutils.get_other_obs(obs),
                    self.storage.get_hidden_state(step),
                    self.storage.get_masks(step),
                    step_info,
                )
                if self.args.clip_actions:
                    ac_info.clip_action(*self.ac_tensor)

            next_obs, reward, done, infos = self.envs.step(ac_info.take_action)
            
            reward += ac_info.add_reward
            
            step_log_vals = rutils.agg_ep_log_stats(infos, ac_info.extra)
            
            self.episode_count += sum([int(d) for d in done])
            self.log.collect_step_info(step_log_vals)

            done = torch.tensor(done.reshape(-1, 1), dtype=torch.bool)
            
            # 0818 - 24차원 → 23차원 변환
            obs = obs[:, :-1] 
            next_obs = next_obs[:, :-1] 
            self.storage.insert(obs, next_obs, reward, done, infos, ac_info)

        updater_log_vals = self.updater.update(self.storage, self.args, beginning, t)
        self.storage.after_update()
        return updater_log_vals
    
class DiffATPSettings(RunSettings):
    def get_policy(self):
        return get_setup_dict()[self.base_args.alg][1](self.base_args.env_name, self.base_args)

    # 0818 (s,a,idx) -> (s,a)입력 차원
    def create_runner(self, add_args={}, ray_create=False) -> rlf.Runner:
        """
        Gets the runner used for training.
        """
        policy = self.get_policy()
        algo = self.get_algo()

        args, log = self._sys_setup(add_args, ray_create, algo, policy)
        if args is None:
            return None
        env_interface = self._get_env_interface(args)

        checkpointer = Checkpointer(args)

        alg_env_settings = algo.get_env_settings(args)
        print("args.eval_only:", args.eval_only)
        #import ipdb; ipdb.set_trace()
        # Setup environment
        _, envs = make_vec_envs(
            args.env_name,
            args.seed,
            args.num_processes,
            args.gamma,
            args.device,
            True,
            env_interface,
            args,
            alg_env_settings,
            #set_eval=args.eval_only,
            False
        )

        rutils.pstart_sep()
        print("Action space:", envs.action_space)
        if isinstance(envs.action_space, Box):
            print("Action range:", (envs.action_space.low, envs.action_space.high))
        print("Observation space", envs.observation_space)
        
        ##0818  
        origin_obs_space = envs.observation_space
        
        # PPO는 idx 없이 (s, expert_action) 23차원만 받도록 수정
        ppo_obs_space = gym.spaces.Box(
            low=origin_obs_space.low[:-1],   
            high=origin_obs_space.high[:-1],
            dtype=origin_obs_space.dtype
        )
        print("Modified PPO Observation space", ppo_obs_space)
        rutils.pend_sep()

        # Setup policy
        policy_args = (ppo_obs_space, envs.action_space, args)
        policy.init(*policy_args)
        policy = policy.to(args.device)
        policy.watch(log)
        policy.set_env_ref(envs)

        # Setup algo
        algo.set_get_policy(self.get_policy, policy_args)
        algo.set_env_ref(envs)
        #import ipdb; ipdb.set_trace()
        algo.init(policy, args)

        # Setup storage buffer
        storage = algo.get_storage_buffer(policy, envs, args)
        for ik, get_shape in alg_env_settings.include_info_keys:
            storage.add_info_key(ik, get_shape(envs))
        storage.to(args.device)
        #0818
        full_obs = envs.reset()  # 24차원
        ppo_obs = full_obs[:, :-1]  # 23차원 (idx 제거)
        storage.init_storage(ppo_obs)
        storage.set_traj_done_callback(algo.on_traj_finished)

        simple_env = ['Sine-v0', 'SCurve-v0', 'Dalmatian-v0', 'Triangle-v0', 'Triangle-v2', 'Rectangle-v0', 'Rectangle-v1']

        #0818 - DiffATP 전용 Runner 사용
        runner = DiffATPRunner(
            envs, storage, policy, log, env_interface, checkpointer, args, algo
        )

        return runner
    
    def create_traj_saver(self, save_path):
        return GoalTrajSaver(save_path, False)

    def get_algo(self):
        algo = get_setup_dict()[self.base_args.alg][0]
        if isinstance(algo, NestedAlgo) and isinstance(algo.modules[0], BaseILAlgo):
            algo.modules[0].set_transform_dem_dataset_fn(trim_episodes_trans)
        if isinstance(algo, SQIL):
            algo.il_algo.set_transform_dem_dataset_fn(trim_episodes_trans)
        return algo

    def get_logger(self):
        if self.base_args.no_wb:
            return BaseLogger()
        else:
            return WbLogger(should_log_vids=True)

    def get_add_args(self, parser):
        parser.add_argument("--alg")
        parser.add_argument("--env-name")
        parser.add_argument("--gw-img", type=str2bool, default=True)
        parser.add_argument("--no-wb", action="store_true", default=False)
        parser.add_argument("--freeze-policy", type=str2bool, default=False)
        parser.add_argument("--rollout-agent", type=str2bool, default=False)
        parser.add_argument("--hidden-dim", type=int, default=256)
        parser.add_argument("--depth", type=int, default=2)
        parser.add_argument("--ppo-hidden-dim", type=int, default=64)
        parser.add_argument("--ppo-layers", type=int, default=2)

    def import_add(self):
        import goal_prox.envs.fetch
        import goal_prox.envs.goal_check

    def get_add_ray_config(self, config):
        return config if self.base_args.no_wb else get_wb_ray_config(config)

    def get_add_ray_kwargs(self):
        return {} if self.base_args.no_wb else get_wb_ray_kwargs()

if __name__ == "__main__":
    run_policy(DiffATPSettings())

    
# # RolloutStorage 활용
# class PPO_rolloutSample(PPO): 
#     def __init__(self, num_steps, num_processes, obs_space, action_space, args):
#         super().__init__(num_steps, num_processes, obs_space, action_space, args)
#         self.next_obs = {}  # next state (s') 저장 공간 추가
#         for k, space in self.ob_keys.items():
#             self.next_obs[k] = torch.zeros(num_steps, num_processes, *space)

#     def get_storage_buffer(self, policy, envs, args) -> RolloutStorage:
#         return PPO_rolloutSample(
#             args.num_steps,
#             args.num_processes,
#             envs.observation_space,
#             envs.action_space,
#             args,
#         )

#     def insert(self, obs, next_obs, rewards, done, info, ac_info):
#         """
#         기존 RolloutStorage insert()에서 next_obs(s')까지 저장하도록 확장
#         """
#         super().insert(obs, next_obs, rewards, done, info, ac_info)

#         for k in self.ob_keys:
#             if k is None:
#                 self.next_obs[self.step].copy_(next_obs)  # next state 저장
#             else:
#                 self.next_obs[k][self.step].copy_(next_obs[k])  # next state 저장

#     def get_generator(self, advantages=None, num_mini_batch=None, mini_batch_size=None, **kwargs):
#         """
#         next_obs(s')까지 포함하여 데이터 샘플링
#         """
#         for indices in super().get_generator(advantages, num_mini_batch, mini_batch_size, **kwargs):
#             indices["next_state"] = self.next_obs[:-1].view(-1, *self.ob_keys[None])[indices["state"].shape[0]:]
#             yield indices

   
