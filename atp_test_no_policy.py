import gym
import envs
import torch
import numpy as np
from atp_envs import ATPEnv_Multiple

# 브로큰 하프치타 환경
base_env = gym.make("HalfCheetahBroken-v2")

# ATPEnv_Multiple 환경 생성 
atp_envs = ATPEnv_Multiple(base_env, rollout_policy_list = []) 

try:
    obs = atp_envs.reset()
except ValueError:
    print("rollout_policy_list is empty. Running environment without policy.")


for i in range(1000):
		action = atp_envs.action_space.sample()
		n_state, rew, done, info = atp_envs.step(action)
        
print(n_state, action, rew, done, info)
