import sys
import torch
from collections import defaultdict
sys.path.insert(0, "./")

from rlf import run_policy

#### Newly Addeded ####
from drail.main import DrailSettings, get_setup_dict
import diffatp.atp_envs
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

def get_setup_dict():
    return {
        "diffATP": (DiffATP(), get_ppo_policy),  
        "ppo": (PPO(), get_ppo_policy),
    }

class DiffATPSettings(RunSettings):
    def get_policy(self):
        return get_setup_dict()[self.base_args.alg][1](self.base_args.env_name, self.base_args)

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

   
