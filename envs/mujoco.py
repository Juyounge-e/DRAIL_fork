# This file is just to get around a baselines import hack.
# env_type is set based on the final part of the entry_point module name.
# In the regular gym mujoco envs this is 'mujoco'.
# We want baselines to treat these as mujoco envs, so we redirect from here,
# and ensure the registry entries are pointing at this file as well.
from .broken_envs import *
from .target_envs import *
import gym

# HalfCheetahBroken-v2 환경 등록 함수
def register_halfcheetah_broken_env():
    gym.envs.registration.register(
        id="HalfCheetahBroken-v2",
        entry_point="envs.target_envs:HalfCheetahModifiedEnv",
    )