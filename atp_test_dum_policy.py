import gym
import envs
import torch
import numpy as np
from atp_envs import ATPEnv_Multiple

# 브로큰 하프치타 환경
base_env = gym.make("HalfCheetahBroken-v2")

# expert_policy_path = "./expert_datasets/HalfCheetah-v3_expert.pt"

class DummyPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def predict(self, observation, deterministic=True):
        """무작위 행동을 반환하는 더미 정책"""
        random_action = np.random.uniform(-1, 1, self.action_dim)
        return random_action, None  # (행동, None)
    
expert_policy = DummyPolicy(base_env.action_space.shape[0])
rollout_policy = [expert_policy]

# ATPEnv_Multiple 환경 생성 
atp_envs = ATPEnv_Multiple(base_env, rollout_policy) 

obs = atp_envs.reset()

for i in range(1000):
    action, _ = expert_policy.predict(obs) 
    n_state, rew, done, info = atp_envs.step(action)
    obs = n_state

print(n_state, action, rew, done, info)
