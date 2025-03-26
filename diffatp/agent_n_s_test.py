import os
import sys
sys.path.insert(0, "./")

import torch
from rlf.storage.rollout_storage import RolloutStorage
import gym
import numpy as np

class DummyArgs:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.recurrent_policy = False  # feed_forward 사용

args = DummyArgs()

# 더미 환경 생성
env = gym.make("Walker2d-v3")
obs_space = env.observation_space
action_space = env.action_space
num_steps = 5
num_processes = 2

# RolloutStorage 생성
storage = RolloutStorage(num_steps, num_processes, obs_space, action_space, args)

# `insert()` 실행 전에 next_state 확인
for step in range(num_steps):
    obs = torch.randn(obs_space.shape)  
    next_obs = torch.randn(obs_space.shape)  
    action = torch.randn(action_space.shape)  
    reward = torch.randn(1)  
    done = torch.tensor([0])  
    info = {}  
    ac_info = type("ActionInfo", (), {"action": action, "action_log_probs": torch.randn(1), "value": torch.randn(1), "hxs": {}})

    masks, bad_masks = storage.compute_masks(done, [info])

    # `bad_masks` 크기 강제 변환 (원본 코드 수정 없이)
    if bad_masks.shape[-1] == 0:
        bad_masks = torch.ones((num_processes, 1))  

    print(f"DEBUG - step {step}: masks shape = {masks.shape}, bad_masks shape = {bad_masks.shape}")
    print(f"DEBUG - step {step}: next_obs shape = {next_obs.shape}")

    # `insert()` 실행 오류를 무시하고 계속 진행하도록 설정
    try:
        storage.insert(obs, next_obs, reward, done, info, ac_info)
    except RuntimeError as e:
        print(f" insert() 실행 중 오류 발생 (무시하고 진행): {e}")
        continue  # 다음 루프로 계속 진행

# `self.obs`가 s, s'을 올바르게 저장하고 있는지 확인
print(f" FINAL - self.obs.shape: {storage.obs.shape}")
print(f" FINAL - State s shape (self.obs[:-1]): {storage.obs[:-1].shape}")
print(f" FINAL - State s' shape (self.obs[1:]): {storage.obs[1:].shape}")

# `agent_n_state`가 포함되었는지 확인
gen = storage.get_generator(num_mini_batch=1)  # `num_mini_batch`를 명시적으로 설정
for agent_batch in gen:
    if 'next_state' in agent_batch:
        print(f" agent_batch에 next_state 포함")
        print(f' Agent Batch keys: {agent_batch.keys()}')
        print(f" next_state Shape: {agent_batch['next_state'].shape}")
        print(f" state Shape: {agent_batch['state'].shape}")
        print(f" Other_state Shape: {agent_batch['other_state']}")
    else:
        print(" agent_batch에 next_state 없음")
    break