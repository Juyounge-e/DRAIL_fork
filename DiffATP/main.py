import sys
import torch
from collections import defaultdict
sys.path.insert(0, "./")

from rlf import run_policy

#### Newly Addeded ####
from drail.main import DrailSettings, get_setup_dict
import DiffATP.atp_envs

from rlf.storage.transition_storage import TransitionStorage
from rlf.storage.rollout_storage import RolloutStorage
from rlf.algos.on_policy.ppo import PPO
from drail.get_policy import get_ppo_policy

from rlf.algos.il.sqil import SQIL
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.il.base_il import BaseILAlgo
from goal_prox.method.utils import trim_episodes_trans

class DiffATPSettings(DrailSettings):
    def get_add_args(self, parser):
        super().get_add_args(parser)
        # env-name: augmented_MDP

    def get_setup_dict(self):
        setup_dict = get_setup_dict()
        setup_dict["PPO_sample"] = (PPO_rolloutSample(), get_ppo_policy)
        return setup_dict
    
    def get_policy(self):
        setup_dict = self.get_setup_dict()
        policy_func = setup_dict[self.base_args.alg][1]  
        return policy_func(self.base_args.env_name, self.base_args)  

    def get_algo(self):
        setup_dict = self.get_setup_dict()
        algo = setup_dict[self.base_args.alg][0]

        if self.base_args.alg == "PPO_sample":
            print("Using PPO_transitionSample with TransitionStorage")
            algo = PPO_rolloutSample()

        # DrailSettings get_algo() 함수 그대로 가져옴  
        if isinstance(algo, NestedAlgo) and isinstance(algo.modules[0], BaseILAlgo):
            algo.modules[0].set_transform_dem_dataset_fn(trim_episodes_trans)
        if isinstance(algo, SQIL):
            algo.il_algo.set_transform_dem_dataset_fn(trim_episodes_trans)

        return algo
    
    
# RolloutStorage 활용
class PPO_rolloutSample(PPO): 
    def __init__(self, num_steps, num_processes, obs_space, action_space, args):
        super().__init__(num_steps, num_processes, obs_space, action_space, args)
        self.next_obs = {}  # next state (s') 저장 공간 추가
        for k, space in self.ob_keys.items():
            self.next_obs[k] = torch.zeros(num_steps, num_processes, *space)

    def get_storage_buffer(self, policy, envs, args) -> RolloutStorage:
        return PPO_rolloutSample(
            args.num_steps,
            args.num_processes,
            envs.observation_space,
            envs.action_space,
            args,
        )

    def insert(self, obs, next_obs, rewards, done, info, ac_info):
        """
        기존 RolloutStorage insert()에서 next_obs(s')까지 저장하도록 확장
        """
        super().insert(obs, next_obs, rewards, done, info, ac_info)

        for k in self.ob_keys:
            if k is None:
                self.next_obs[self.step].copy_(next_obs)  # next state 저장
            else:
                self.next_obs[k][self.step].copy_(next_obs[k])  # next state 저장

    def get_generator(self, advantages=None, num_mini_batch=None, mini_batch_size=None, **kwargs):
        """
        next_obs(s')까지 포함하여 데이터 샘플링
        """
        for indices in super().get_generator(advantages, num_mini_batch, mini_batch_size, **kwargs):
            indices["next_state"] = self.next_obs[:-1].view(-1, *self.ob_keys[None])[indices["state"].shape[0]:]
            yield indices

   
if __name__ == "__main__":
    run_policy(DiffATPSettings())