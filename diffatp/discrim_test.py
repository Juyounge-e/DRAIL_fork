import os
import sys
sys.path.insert(0, "./")

import torch
from rlf.algos.il.base_il import BaseILAlgo
from drail.drail import DRAILDiscrim
import os

class DummyArgs:
    def __init__(self):
        self.traj_load_path = "/home/imitation/DRAIL/data/traj/Walker2d-v3/0304/34-W-1-08-ppo/trajs.pt"  
        self.traj_batch_size = 64
        self.traj_frac = 1.0
        self.traj_val_ratio = 0.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cwd = os.getcwd()  # 현재 디렉토리 사용
        self.traj_viz = False

args = DummyArgs()
drail_discrim = DRAILDiscrim()
drail_discrim._load_expert_data(None, args)  # 전문가 데이터 로드

for expert_batch in drail_discrim.expert_train_loader:
    if 'next_state' in expert_batch:
        print("🔎 expert_batch keys:", expert_batch.keys())
        print(f"next_state Shape: {expert_batch['next_state'].shape}")
        print(f" state Shape: {expert_batch['state'].shape}")
        print(f"next_state Shape: {expert_batch['next_state']}")
        print(f" state Shape: {expert_batch['state']}")
    else:
        print("expert_batch에 next_state 없음")
    break  #