# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling D4RL or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .. import offline_env
from gym.envs.mujoco import HalfCheetahEnv, AntEnv, HopperEnv, Walker2dEnv
from ..utils.wrappers import NormalizedBoxEnv

class OfflineAntEnv(AntEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        AntEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHopperEnv(HopperEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HopperEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineHalfCheetahEnv(HalfCheetahEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HalfCheetahEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

class OfflineWalker2dEnv(Walker2dEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        Walker2dEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

    # def set_noise_ratio(self, noise_ratio, goal_noise_ratio):
    #     self.reset_noise_scale = noise_ratio


def get_ant_env(**kwargs):
    return NormalizedBoxEnv(OfflineAntEnv(**kwargs))

def get_cheetah_env(**kwargs):
    return NormalizedBoxEnv(OfflineHalfCheetahEnv(**kwargs))

def get_hopper_env(**kwargs):
    return NormalizedBoxEnv(OfflineHopperEnv(**kwargs))

def get_walker_env(**kwargs):
    return NormalizedBoxEnv(OfflineWalker2dEnv(**kwargs))

if __name__ == '__main__':
    """Example usage of these envs"""
    pass
