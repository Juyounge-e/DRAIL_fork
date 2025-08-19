"""
DiffATP 전용 PPO 구현
- observation space에서 idx 차원을 제거하여 처리
- (s, expert_action, idx) 24차원 → (s, expert_action) 23차원
"""

import gym
from rlf.algos.on_policy.ppo import PPO
from rlf.storage.rollout_storage import RolloutStorage


class DiffATPPPO(PPO):
    """
    DiffATP용 PPO: observation space에서 idx 차원을 제거하여 23차원으로 처리
    
    1. Storage Buffer를 23차원으로 생성
    2. 환경의 24차원 observation에서 idx 제거
    """
    
    def get_storage_buffer(self, policy, envs, args):
        """
        23차원 observation space로 RolloutStorage 생성
        원본 envs.observation_space (24차원)에서 idx(마지막 차원) 제거
        """
        # 원본 24차원에서 idx(마지막 차원) 제거하여 23차원으로 수정
        original_obs_space = envs.observation_space
        ppo_obs_space = gym.spaces.Box(
            low=original_obs_space.low[:-1],
            high=original_obs_space.high[:-1],
            dtype=original_obs_space.dtype
        )
        
        print(f"🔧 DiffATPPPO: Original obs space: {original_obs_space.shape}")
        print(f"🔧 DiffATPPPO: Modified obs space: {ppo_obs_space.shape}")
        
        return RolloutStorage(
            args.num_steps, 
            args.num_processes,
            ppo_obs_space,  # 23차원 observation space 사용
            envs.action_space, 
            args,
            hidden_states=policy.get_storage_hidden_states()
        )
    