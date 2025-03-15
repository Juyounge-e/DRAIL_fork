import gym
import envs
import torch
import numpy as np
from atp_envs import ATPEnv_Multiple

# 하프치타 환경
base_env = gym.make("HalfCheetah-v3")
expert_policy_path = "./data/traj/HalfCheetah-v3/212-HC-1-4K-ppo/trajs.pt"

class ExpertPolicy:
    def __init__(self, model_path):
        self.data = torch.load(model_path) 
        self.obs = self.data['obs']  # 상태 데이터
        self.actions = self.data['actions']  # 행동 데이터

        self.num_samples = self.obs.shape[0]  # 샘플 수
        self.idx = 0  # 샘플 인덱스

    def predict(self, observation, deterministic=True):
        """SAC의 predict()와 동일한 인터페이스 유지"""
        action = self.actions[self.idx]  # 미리 저장된 전문가 행동 사용
        self.idx = (self.idx + 1) % self.num_samples  # 인덱스 업데이트
        action = action.cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action) # 텐서 to 배열
        return action, None 
    
            with self.train_ctx():
                ac_info = self.policy.get_action(
                    utils.get_def_obs(obs, self.args.policy_ob_key),
                    utils.get_other_obs(obs),
                    self.storage.get_hidden_state(step),
                    self.storage.get_masks(step),
                    step_info,
                )
                if self.args.clip_actions:
                    ac_info.clip_action(*self.ac_tensor)

            next_obs, reward, done, infos = self.envs.step(ac_info.take_action)
        

expert_policy = ExpertPolicy(expert_policy_path)
rollout_policy = [expert_policy]

# ATPEnv_Multiple 환경 생성 
atp_env = ATPEnv_Multiple(base_env, rollout_policy) 

obs = atp_env.reset()

for i in range(1000):
    action, _ = atp_env.rollout_policy.predict(obs)  
    n_state, rew, done, info = atp_env.step(action)
    
    if done:  
        obs = atp_env.reset() 

print(n_state, action, rew, done, info)
