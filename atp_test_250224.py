import gym
# import envs
import torch
import numpy as np
# from atp_envs import ATPEnv_Multiple

from rlf.policies import DistActorCritic
from rlf.rl.model import MLPBasic

from rlf.policies.base_policy import StepInfo
from rlf.rl import utils



# 하프치타 환경
base_env = gym.make("Walker2d-v3")
expert_policy_path = "/home/imitation/DRAIL-fork/data/trained_models/Walker2d-v3/26-W-1-A4-ppo/model_3051.pt"

# Expert policy load and act test
num_processes = 1
device = torch.device("cpu")
expert_policy_state = torch.load(expert_policy_path)
print(expert_policy_state)
# expert_policy = DistActorCritic()
expert_policy = DistActorCritic(
        get_actor_fn=lambda _, i_shape: MLPBasic(
        i_shape[0], hidden_size=256, num_layers=2
        ),
        get_critic_fn=lambda _, i_shape, asp: MLPBasic(
            i_shape[0], hidden_size=256, num_layers=2
        ),
    )#

expert_policy = (base_env.observation_space, base_env.action_space, args)
expert_policy.init(*policy_args)
expert_policy = expert_policy.to(device)
expert_policy.set_env_ref(base_env)
expert_policy.load_state_dict(expert_policy_state["policy"])

###########################################################################

hidden_states = {}
for k, dim in expert_policy.get_storage_hidden_states().items():
    hidden_states[k] = torch.zeros(num_processes, dim).to(device)
eval_masks = torch.zeros(num_processes, 1, device=device)
policy.eval() # Evaluation mode

obs = base_env.reset()
for i in range(1000):
    step_info = StepInfo(None, None, True) # from rlf.policies.base_policy import StepInfo
    with torch.no_grad():
        act_obs = utils.ob_to_np(obs) # from rlf.rl.utils.ob_to_np
        act_obs = utils.ob_to_tensor(act_obs, device)
        ac_info = expert_policy.get_action(
            utils.get_def_obs(act_obs),
            utils.get_other_obs(obs),
            hidden_states,
            eval_masks,
            step_info,
        )
        
        hidden_states = ac_info.hxs

    # Observe reward and next obs
    next_obs,  _, done, infos = base_env.step(ac_info.take_action)

    eval_masks = torch.tensor(
        [[0.0] if done_ else [1.0] for done_ in done],
        dtype=torch.float32,
        device=device,
    )
    obs = next_obs

# rollout_policy = [expert_policy]

# # ATPEnv_Multiple 환경 생성 
# atp_env = ATPEnv_Multiple(base_env, rollout_policy) 

# obs = atp_env.reset()

# for i in range(1000):
#     action, _ = atp_env.rollout_policy.predict(obs)  
#     n_state, rew, done, info = atp_env.step(action)
    
#     if done:  
#         obs = atp_env.reset() 

print(n_state, action, rew, done, info)
