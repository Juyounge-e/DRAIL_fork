import gym
from gym import spaces
import numpy as np
import torch

from rlf.envs.env_interface import EnvInterface
from rlf.envs.env_interface import register_env_interface, get_env_interface
from rlf.policies.base_policy import get_empty_step_info
import rlf.rl.utils as rutils
from rlf.baselines.vec_env.dummy_vec_env import DummyVecEnv

from DiffATP.main import DiffATPSettings


class ATPEnv_Multiple(gym.Wrapper):
    """
    Defines augmented MDP for learning action transformation policy
    """
    def __init__(self,
                 env, # halfcheetah 
                 rollout_policy_list, # expert policy
                #  seed=None,
                 dynamic_prob=False,
                 deter_rollout=False,
                 args = None
                 ):
        super(ATPEnv_Multiple, self).__init__(env)
        # env.seed(seed)
        self.rollout_policy_list = rollout_policy_list
        self.deter_rollout = deter_rollout

        # Set range of transformed action
        low = np.concatenate((self.env.observation_space.low,
                              self.env.action_space.low,
                              np.zeros(len(self.rollout_policy_list)))
                             )
        high = np.concatenate((self.env.observation_space.high,
                               self.env.action_space.high,
                               np.zeros(len(self.rollout_policy_list)))
                              )

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.env_max_act = (self.env.action_space.high - self.env.action_space.low) / 2

        max_act = (self.env.action_space.high - self.env.action_space.low)
        self.action_space = spaces.Box(-max_act, max_act, dtype=np.float32)

        # These are set when reset() is called
        self.latest_obs = None
        self.latest_act = None

        self.rollout_dist = np.ones(len(self.rollout_policy_list)) * 1/len(self.rollout_policy_list)
        self.num_selection = np.zeros(len(self.rollout_policy_list))
        self.dynamic_prob = dynamic_prob
        if dynamic_prob:
            self.rew_list = []
            self.rew_threshold = self.spec.reward_threshold

        self.args = args

    def reset(self, **kwargs):
        """Reset function for the wrapped environment"""
        self.latest_obs = self.env.reset(**kwargs)

        if self.dynamic_prob and len(self.rew_list) > 0:
            c_returns = np.sum(self.rew_list)
            self.rew_list = []
            idx_to_update = int(self.num_selection[self.idx] % np.shape(self.epi_returns_g)[1])
            self.epi_returns_g[self.idx, idx_to_update] = c_returns

            avg_epi_return_g = np.average(self.epi_returns_g, axis=1)
            gap = np.abs(self.epi_returns_d - avg_epi_return_g)
            prob_dist = np.exp(gap/self.rew_threshold)
            self.rollout_dist = prob_dist / np.sum(prob_dist)
            if np.sum(self.num_selection)%1000==0:
                print(self.rollout_dist)

        # Choose rollout policy
        self.idx = np.random.choice(len(self.rollout_policy_list), p = self.rollout_dist)
        self.rollout_policy = self.rollout_policy_list[self.idx]
        self.idx_obs = np.eye(len(self.rollout_policy_list))[self.idx]
        self.num_selection[self.idx] += 1


        self.hidden_states = {}
        for k, dim in self.rollout_policy.get_storage_hidden_states().items():
            self.hidden_states[k] = torch.zeros(self.args.num_processes, dim).to(self.args.device)
        self.eval_masks = torch.zeros(self.args.num_processes, 1, device=self.args.device)
        self.rollout_policy.eval()

        step_info = get_empty_step_info()
        with torch.no_grad():
            # act_obs = rutils.ob_to_np(self.latest_obs) # obfilter 제외, 여기서 self.latest_obs는 np
            act_obs = np.array([self.latest_obs])
            act_obs = rutils.ob_to_tensor(act_obs, self.args.device).float() # 여기서만 type 변환 추가 진행
            ac_info = self.rollout_policy.get_action(
                rutils.get_def_obs(act_obs),
                rutils.get_other_obs(self.latest_obs),
                self.hidden_states,
                self.eval_masks,
                step_info,
            ) # basepolicy에서 Deterministic option 가능한지 확인
            self.hidden_states = ac_info.hxs

        self.latest_act = ac_info.take_action
        self.latest_act = self.latest_act.cpu().numpy()[0]
        # Return (s,a)
        # return np.append(self.latest_obs, self.latest_act)
        return np.concatenate((self.latest_obs, self.latest_act, self.idx_obs))

    def init_selection_prob(self, epi_returns_d, epi_returns_g):
        self.epi_returns_d = epi_returns_d
        self.epi_returns_g = epi_returns_g

    def step(self, action):
        """
        Step function for the wrapped environment
        """
        # input action is the delta transformed action for this Environment
        transformed_action = action + self.latest_act # 기존 액션(latest_act) + 변화량
        transformed_action = np.clip(transformed_action, - self.env_max_act, self.env_max_act) # 기준을 넘자 않도록 

        sim_next_state, sim_rew, sim_done, info = self.env.step(transformed_action)
        # self.eval_masks = torch.tensor(
        #     [[0.0] if done_ else [1.0] for done_ in sim_done],
        #     dtype=torch.float32,
        #     device=self.device,
        # )
        self.eval_masks = torch.tensor(
            [0.0 if sim_done else 1.0],
            dtype=torch.float32,
            device=self.args.device,
        )

        if self.dynamic_prob:
            self.rew_list.append(sim_rew)

        info['transformed_action'] = transformed_action # 기록

        # get target policy action
        step_info = get_empty_step_info()
        with torch.no_grad():
            act_obs = np.array([self.latest_obs])
            act_obs = rutils.ob_to_tensor(act_obs, self.args.device).float() # 여기서만 type 변환 추가 진행
            ac_info = self.rollout_policy.get_action(
                rutils.get_def_obs(act_obs),
                rutils.get_other_obs(self.latest_obs),
                self.hidden_states,
                self.eval_masks,
                step_info,
            )
            self.hidden_states = ac_info.hxs
        target_policy_action = ac_info.take_action
        target_policy_action = target_policy_action.cpu().numpy()[0]

        # concat_sa = np.append(sim_next_state, target_policy_action)
        concat_sa = np.concatenate((sim_next_state, target_policy_action, self.idx_obs)) # 기존 action,  expert action, expert 번호 

        self.latest_obs = sim_next_state
        self.latest_act = target_policy_action

        return concat_sa, sim_rew, sim_done, info 

    def close(self):
        self.env.close()

class ATPEnvInterface(EnvInterface):
    def get_add_args(self, parser):
        parser.add_argument("--src-env-name", type=str, default = None) # src-env-name: e.g. Walker-v2
        parser.add_argument("--rollout-policy-path", type=str, default = None)
        # parser.add_argument("--dynamic_prob", default = "store_true")
        parser.add_argument("--deter_rollout", default = "store_true")

    def create_from_id(self, env_id):
        # Note: argument에서 seed 뺐음.
        self.src_env = gym.make(self.args.src_env_name)
        rollout_policy_list = []
        path_list = self.args.rollout_policy_path.split(", ")
        print(path_list)
        for load_path in path_list:
            print("load_path: ", load_path)
            policy = self.load_rollout_policy(load_path)
            if self.args.deter_rollout:
                policy.args.deterministic_policy = True # Working하는지 확인 필요
            rollout_policy_list.append(policy)
        return ATPEnv_Multiple(self.src_env, rollout_policy_list, deter_rollout=self.args.deter_rollout, args = self.args)

    def load_rollout_policy(self, rollout_policy_path, seed=None):
        policy_state = torch.load(rollout_policy_path)
        policy = get_setup_dict()["ppo"][1](self.args.src_env_name, self.args)
        policy_args = (self.src_env.observation_space, self.src_env.action_space, self.args)
        policy.init(*policy_args)
        policy = policy.to(self.args.device)
        policy.set_env_ref(self.src_env)
        policy.load_state_dict(policy_state["policy"])
        policy.eval()
        return policy

# Match any version
register_env_interface("augmented_MDP-v0", ATPEnvInterface)

# class ATPEnv_Multiple(gym.Wrapper):
#     """
#     Defines augmented MDP for learning action transformation policy
#     """
#     def __init__(self,
#                  env, # halfcheetah 
#                  rollout_policy_list, # expert policy
#                  seed=None,
#                  dynamic_prob=False,
#                  deter_rollout=False,
#                  ):
#         super(ATPEnv_Multiple, self).__init__(env)
#         env.seed(seed)
#         self.rollout_policy_list = rollout_policy_list
#         self.deter_rollout = deter_rollout

#         # Set range of transformed action
#         low = np.concatenate((self.env.observation_space.low,
#                               self.env.action_space.low,
#                               np.zeros(len(self.rollout_policy_list)))
#                              )
#         high = np.concatenate((self.env.observation_space.high,
#                                self.env.action_space.high,
#                                np.zeros(len(self.rollout_policy_list)))
#                               )

#         self.observation_space = spaces.Box(low, high, dtype=np.float32)
#         self.env_max_act = (self.env.action_space.high - self.env.action_space.low) / 2

#         max_act = (self.env.action_space.high - self.env.action_space.low)
#         self.action_space = spaces.Box(-max_act, max_act, dtype=np.float32)

#         # These are set when reset() is called
#         self.latest_obs = None
#         self.latest_act = None

#         self.rollout_dist = np.ones(len(self.rollout_policy_list)) * 1/len(self.rollout_policy_list)
#         self.num_selection = np.zeros(len(self.rollout_policy_list))
#         self.dynamic_prob = dynamic_prob
#         if dynamic_prob:
#             self.rew_list = []
#             self.rew_threshold = self.spec.reward_threshold

#     def reset(self, **kwargs):
#         """Reset function for the wrapped environment"""
#         self.latest_obs = self.env.reset(**kwargs)

#         if self.dynamic_prob and len(self.rew_list) > 0:
#             c_returns = np.sum(self.rew_list)
#             self.rew_list = []
#             idx_to_update = int(self.num_selection[self.idx] % np.shape(self.epi_returns_g)[1])
#             self.epi_returns_g[self.idx, idx_to_update] = c_returns

#             avg_epi_return_g = np.average(self.epi_returns_g, axis=1)
#             gap = np.abs(self.epi_returns_d - avg_epi_return_g)
#             prob_dist = np.exp(gap/self.rew_threshold)
#             self.rollout_dist = prob_dist / np.sum(prob_dist)
#             if np.sum(self.num_selection)%1000==0:
#                 print(self.rollout_dist)

#         # Choose rollout policy
#         self.idx = np.random.choice(len(self.rollout_policy_list), p = self.rollout_dist)
#         self.rollout_policy = self.rollout_policy_list[self.idx]
#         self.idx_obs = np.eye(len(self.rollout_policy_list))[self.idx]
#         self.num_selection[self.idx] += 1

#         self.latest_act, _ = self.rollout_policy.predict(self.latest_obs, deterministic=self.deter_rollout)

#         # Return (s,a)
#         # return np.append(self.latest_obs, self.latest_act)
#         return np.concatenate((self.latest_obs, self.latest_act, self.idx_obs))

#     def init_selection_prob(self, epi_returns_d, epi_returns_g):
#         self.epi_returns_d = epi_returns_d
#         self.epi_returns_g = epi_returns_g

#     def step(self, action):
#         """
#         Step function for the wrapped environment
#         """
#         # input action is the delta transformed action for this Environment
#         transformed_action = action + self.latest_act # 기존 액션(latest_act) + 변화량
#         transformed_action = np.clip(transformed_action, - self.env_max_act, self.env_max_act) # 기준을 넘자 않도록 

#         sim_next_state, sim_rew, sim_done, info = self.env.step(transformed_action)
#         if self.dynamic_prob:
#             self.rew_list.append(sim_rew)

#         info['transformed_action'] = transformed_action # 기록

#         # get target policy action
#         target_policy_action, _ = self.rollout_policy.predict(sim_next_state, deterministic=self.deter_rollout) # observation이 들어갔을때 expert policy가 어떻게 판단하는지 

#         # concat_sa = np.append(sim_next_state, target_policy_action)
#         concat_sa = np.concatenate((sim_next_state, target_policy_action, self.idx_obs)) # 기존 action,  expert action, expert 번호 

#         self.latest_obs = sim_next_state
#         self.latest_act = target_policy_action

#         return concat_sa, sim_rew, sim_done, info 

#     def close(self):
#         self.env.close()

# # class GroundedEnv(gym.ActionWrapper):
#     """
#     Defines the grounded environment
#     """
#     def __init__(self,
#                  env,
#                  action_tf_policy,
#                  use_deterministic=True,
#                  ):
#         super(GroundedEnv, self).__init__(env)
#         if isinstance(action_tf_policy, list):
#             self.num_simul = len(action_tf_policy)
#             idx = np.random.randint(0,self.num_simul)
#             self.atp_list = action_tf_policy
#             self.action_tf_policy = self.atp_list[idx]
#         else:
#             self.num_simul = 1
#             self.action_tf_policy = action_tf_policy

#         self.transformed_action_list = []
#         self.raw_actions_list = []

#         self.latest_obs = None
#         self.time_step_counter = 0
#         self.high = env.action_space.high
#         self.low = env.action_space.low

#         max_act = (self.high - self.low)
#         self.transformed_action_space = spaces.Box(-max_act, max_act, dtype=np.float32)
#         self.use_deterministic = use_deterministic

#     def reset(self, **kwargs):
#         self.latest_obs = self.env.reset(**kwargs)
#         self.time_step_counter = 0
#         if self.num_simul > 1:
#             idx = np.random.randint(0, self.num_simul)
#             self.action_tf_policy = self.atp_list[idx]
#         return self.latest_obs

#     def step(self, action):
#         self.time_step_counter += 1

#         concat_sa = np.append(self.latest_obs, action)

#         delta_transformed_action, _ = self.action_tf_policy.predict(concat_sa, deterministic=self.use_deterministic)
#         transformed_action = action + delta_transformed_action
#         transformed_action = np.clip(transformed_action, self.low, self.high)

#         self.latest_obs, rew, done, info = self.env.step(transformed_action)

#         info['transformed_action'] = transformed_action
#         if self.time_step_counter <= 1e4:
#             self.transformed_action_list.append(transformed_action)
#             self.raw_actions_list.append(action)

#         # change the reward to be a function of the input action and
#         # not the transformed action
#         if 'Hopper' in self.env.unwrapped.spec.id:
#             rew = rew - 1e-3 * np.square(action).sum() + 1e-3 * np.square(transformed_action).sum()
#         elif 'HalfCheetah' in self.env.unwrapped.spec.id:
#             rew = rew - 0.1 * np.square(action).sum() + 0.1 * np.square(transformed_action).sum()
#         return self.latest_obs, rew, done, info

#     def reset_saved_actions(self):
#         self.transformed_action_list = []
#         self.raw_actions_list = []

#     def plot_action_transformation(
#             self,
#             expt_path=None,
#             show_plot=False,
#             max_points=3000,
#             true_transformation=None):
#         """Graphs transformed actions vs input actions"""
#         num_action_space = self.env.action_space.shape[0]
#         action_low = self.env.action_space.low[0]
#         action_high = self.env.action_space.high[0]

#         self.raw_actions_list = np.asarray(self.raw_actions_list)
#         self.transformed_action_list = np.asarray(self.transformed_action_list)

#         opt_gap = None
#         if true_transformation is not None:
#             if true_transformation == 'Broken':
#                 true_array = np.copy(self.raw_actions_list)
#                 true_array[:, 0] = np.zeros_like(true_array[:, 0])
#                 opt_gap = np.mean(np.linalg.norm(true_array - self.transformed_action_list, axis=1))
#             else:
#                 true_array = np.ones_like(self.transformed_action_list) * true_transformation
#                 opt_gap = np.mean(np.linalg.norm(true_array - (self.transformed_action_list - self.raw_actions_list), axis=1))

#         mean_delta = np.mean(np.abs(self.raw_actions_list - self.transformed_action_list))
#         max_delta = np.max(np.abs(self.raw_actions_list - self.transformed_action_list))
#         print("Mean delta transformed_action: ", mean_delta)
#         print("Max:", max_delta)
#         # Reduce sample size
#         index = np.random.choice(np.shape(self.raw_actions_list)[0], max_points, replace=False)
#         self.raw_actions_list = self.raw_actions_list[index]
#         self.transformed_action_list = self.transformed_action_list[index]

#         colors = ['go', 'bo', 'ro', 'mo', 'yo', 'ko', 'go', 'bo', 'ro', 'mo', 'yo', 'ko']

#         if num_action_space > len(colors):
#             print("Unsupported Action space shape.")
#             return

#         # plotting the data points starts here
#         fig = plt.figure(figsize=(int(10*num_action_space), 8))
#         plt.rcParams['font.size'] = '24'
#         for act_num in range(num_action_space):
#             ax = fig.add_subplot(1, num_action_space, act_num+1)
#             ax.plot(self.raw_actions_list[:, act_num], self.transformed_action_list[:, act_num], colors[act_num], alpha=1, markersize = 2)
#             if true_transformation is not None:
#                 if true_transformation == 'Broken':
#                     if act_num == 0:
#                         ax.plot([action_low, action_high], [0, 0], 'k-')
#                     else:
#                         ax.plot([action_low, action_high], [action_low, action_high], 'k-')
#                 else:
#                     ax.plot([action_low, action_high], [-1.5, 4.5], 'k-')
#             ax.title.set_text('Action Dimension '+ str(act_num+1))
#             if act_num == 0:
#                 ax.set_ylabel('Transformed action')
#             ax.set(xlabel = 'Original action', xlim=[action_low, action_high], ylim=[action_low, action_high])

#         plt.savefig(expt_path)
#         if show_plot: plt.show()
#         plt.close()

#         return mean_delta, max_delta, opt_gap

#     def test_grounded_environment(self,
#                                   expt_path,
#                                   target_policy=None,
#                                   random=True,
#                                   true_transformation=None,
#                                   num_steps=2048,
#                                   deter_target=False):
#         """Tests the grounded environment for action transformation"""
#         print("TESTING GROUNDED ENVIRONMENT")
#         self.reset_saved_actions()
#         obs = self.reset()
#         time_step_count = 0
#         for _ in range(num_steps):
#             time_step_count += 1
#             if not random:
#                 action, _ = target_policy.predict(obs, deterministic=deter_target)
#             else:
#                 action = self.action_space.sample()
#             obs, _, done, _ = self.step(action)
#             if done:
#                 obs = self.reset()
#                 done = False

#         _, _, opt_gap = self.plot_action_transformation(expt_path=expt_path, max_points=num_steps, true_transformation=true_transformation)
#         self.reset_saved_actions()

#         return opt_gap

#     def close(self):
#         self.env.close()