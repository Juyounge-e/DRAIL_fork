import torch

trajs = torch.load("/home/imitation/DRAIL/data/traj/Walker2d-v3/0304/34-W-1-08-ppo/trajs.pt")

print(trajs.keys())  # 저장된 키 확인
print(f"obs shape: {trajs['obs'].shape}")      # 상태 (s)
print(f"actions shape: {trajs['actions'].shape}")  # 행동 (a)
print(f"done: {trajs['done'].shape}") # 종료 여부
print(f"next_obs shape: {trajs['next_obs'].shape}") # 다음 상태 (s')


