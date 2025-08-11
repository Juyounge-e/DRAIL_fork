import gym
from gym import spaces
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import os

import rlf.rl.utils as rutils
from rlf.policies.base_policy import get_empty_step_info


class GroundedEnv(gym.Wrapper):
    """
    Defines the grounded environment
    """
    def __init__(self,
                env,
                action_tf_policy,
                args,
                use_deterministic=True,
                ):
        super(GroundedEnv, self).__init__(env)
        self.args = args
        
        if isinstance(action_tf_policy, list):
            self.num_simul = len(action_tf_policy)
            idx = np.random.randint(0, self.num_simul)
            self.atp_list = action_tf_policy
            self.action_tf_policy = self.atp_list[idx]
        else:
            self.num_simul = 1
            self.action_tf_policy = action_tf_policy

        self.transformed_action_list = []
        self.raw_actions_list = []

        self.latest_obs = None
        self.time_step_counter = 0
        self.high = env.action_space.high
        self.low = env.action_space.low

        max_act = (self.high - self.low)
        self.transformed_action_space = spaces.Box(-max_act, max_act, dtype=np.float32)
        self.use_deterministic = use_deterministic
        
        # Initialize policy states for rl-toolkit 
        self.hidden_states = {}
        if hasattr(self.action_tf_policy, 'get_storage_hidden_states'):
            for k, dim in self.action_tf_policy.get_storage_hidden_states().items():
                self.hidden_states[k] = torch.zeros(1, dim).to(self.args.device)
        
        self.eval_masks = torch.ones(1, 1, device=self.args.device)

    def reset(self, **kwargs):
        self.latest_obs = self.env.reset(**kwargs)
        self.time_step_counter = 0
        if self.num_simul > 1:
            idx = np.random.randint(0, self.num_simul)
            self.action_tf_policy = self.atp_list[idx]
        return self.latest_obs

    def step(self, action):
        self.time_step_counter += 1
        concat_sa = np.append(self.latest_obs, action)
        delta_transformed_action, _ = self.action_tf_policy.predict(
            concat_sa, deterministic=self.use_deterministic
        )

        transformed_action = action + delta_transformed_action
        transformed_action = np.clip(transformed_action, self.low, self.high)

        self.latest_obs, rew, done, info = self.env.step(transformed_action)

        info['transformed_action'] = transformed_action
        if self.time_step_counter <= 1e4:
            self.transformed_action_list.append(transformed_action)
            self.raw_actions_list.append(action)

        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'spec'):
            if 'Hopper' in self.env.unwrapped.spec.id:
                rew = rew - 1e-3 * np.square(action).sum() + 1e-3 * np.square(transformed_action).sum()
            elif 'HalfCheetah' in self.env.unwrapped.spec.id:
                rew = rew - 0.1 * np.square(action).sum() + 0.1 * np.square(transformed_action).sum()

        return self.latest_obs, rew, done, info

    def reset_saved_actions(self):
        self.transformed_action_list = []
        self.raw_actions_list = []

    def plot_action_transformation(self,
                                expt_path=None,
                                show_plot=False,
                                max_points=3000,
                                true_transformation=None,
                                logger=None):
        """Graphs transformed actions vs input actions"""     
        num_action_space = self.env.action_space.shape[0]
        action_low = self.env.action_space.low[0]
        action_high = self.env.action_space.high[0]

        raw_actions = np.asarray(self.raw_actions_list)
        transformed_actions = np.asarray(self.transformed_action_list)

        opt_gap = None
        if true_transformation is not None:
            if true_transformation == 'Broken':
                true_array = np.copy(raw_actions)
                true_array[:, 0] = np.zeros_like(true_array[:, 0])
                opt_gap = np.mean(np.linalg.norm(true_array - transformed_actions, axis=1))
            else:
                true_array = np.ones_like(transformed_actions) * true_transformation
                opt_gap = np.mean(np.linalg.norm(true_array - (transformed_actions - raw_actions), axis=1))

        mean_delta = np.mean(np.abs(raw_actions - transformed_actions))
        max_delta = np.max(np.abs(raw_actions - transformed_actions))
        
        # Log metrics through project logging system
        if logger is not None:
            try:
                logger.log_vals({
                    'grounded_env/mean_delta': mean_delta,
                    'grounded_env/max_delta': max_delta,
                    'grounded_env/opt_gap': opt_gap if opt_gap is not None else 0.0,
                    'grounded_env/num_samples': len(raw_actions)
                }, self.time_step_counter)
            except (AttributeError, Exception) as e:
                print(f"Warning: Logger error (skipping): {e}")
                # Continue without logging

        print(f"Mean delta transformed_action: {mean_delta}")
        print(f"Max delta: {max_delta}")
        
        if len(raw_actions) > max_points:
            index = np.random.choice(len(raw_actions), max_points, replace=False)
            raw_actions = raw_actions[index]
            transformed_actions = transformed_actions[index]

        colors = ['go', 'bo', 'ro', 'mo', 'yo', 'ko', 'go', 'bo', 'ro', 'mo', 'yo', 'ko']

        if num_action_space > len(colors):
            print("Unsupported Action space shape.")
            return mean_delta, max_delta, opt_gap

        # Create plot
        fig = plt.figure(figsize=(int(10*num_action_space), 8))
        plt.rcParams['font.size'] = '16'  
        
        for act_num in range(num_action_space):
            ax = fig.add_subplot(1, num_action_space, act_num+1)
            ax.plot(raw_actions[:, act_num], transformed_actions[:, act_num], 
                colors[act_num], alpha=0.7, markersize=2)
            
            if true_transformation is not None:
                if true_transformation == 'Broken':
                    if act_num == 0:
                        ax.plot([action_low, action_high], [0, 0], 'k-', linewidth=2)
                    else:
                        ax.plot([action_low, action_high], [action_low, action_high], 'k-', linewidth=2)
                else:
                    ax.plot([action_low, action_high], [-1.5, 4.5], 'k-', linewidth=2)
                    
            ax.set_title(f'Action Dimension {act_num+1}')
            if act_num == 0:
                ax.set_ylabel('Transformed action')
            ax.set_xlabel('Original action')
            ax.set_xlim([action_low, action_high])
            ax.set_ylim([action_low, action_high])
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        if expt_path is not None:
            plt.savefig(expt_path, dpi=150, bbox_inches='tight')
            # Log image if logger supports it
            if logger is not None and hasattr(logger, 'log_image'):
                logger.log_image('grounded_env/action_transformation', expt_path, self.time_step_counter)
        
        if show_plot:
            plt.show()
        plt.close()

        return mean_delta, max_delta, opt_gap

    def test_grounded_environment(self,
                                expt_path,
                                target_policy=None,
                                random=True,
                                true_transformation=None,
                                num_steps=2048,
                                deter_target=False,
                                logger=None):
        """Tests the grounded environment for action transformation"""
        print("TESTING GROUNDED ENVIRONMENT")
        self.reset_saved_actions()
        obs = self.reset()
        time_step_count = 0
        
        # Initialize target policy states if needed
        if target_policy is not None:
            target_hidden_states = {}
            if hasattr(target_policy, 'get_storage_hidden_states'):
                for k, dim in target_policy.get_storage_hidden_states().items():
                    target_hidden_states[k] = torch.zeros(1, dim).to(self.args.device)
            target_eval_masks = torch.ones(1, 1, device=self.args.device)
        
        for _ in range(num_steps):
            time_step_count += 1
            
            if not random and target_policy is not None:
                # Use rl-toolkit policy interface (predict을 변환)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.args.device)
                step_info = get_empty_step_info()
                with torch.no_grad():
                    target_policy.eval()
                    ac_info = target_policy.get_action(
                        obs_tensor,
                        None,  # add_state
                        target_hidden_states,
                        target_eval_masks,
                        step_info
                    )
                    target_hidden_states = ac_info.hxs
                    
                action = ac_info.take_action.squeeze(0).cpu().numpy()
            else:
                action = self.action_space.sample()
                
            obs, _, done, _ = self.step(action)
            
            if done:
                obs = self.reset()
                done = False
                # Reset target policy states if needed
                if target_policy is not None:
                    if hasattr(target_policy, 'get_storage_hidden_states'):
                        for k, dim in target_policy.get_storage_hidden_states().items():
                            target_hidden_states[k] = torch.zeros(1, dim).to(self.args.device)
                    target_eval_masks = torch.ones(1, 1, device=self.args.device)

        # Create plots and log results
        _, _, opt_gap = self.plot_action_transformation(
            expt_path=expt_path, 
            max_points=num_steps, 
            true_transformation=true_transformation,
            logger=logger
        )
        
        self.reset_saved_actions()
        return opt_gap

    def close(self):
        self.env.close()

    def get_env_settings(self):
        """Returns environment settings for integration with rl-toolkit"""
        from rlf.envs.env_interface import EnvSettings
        
        settings = EnvSettings()
        
        def get_transformed_action_shape(envs):
            return self.action_space.shape
            
        def get_delta_shape(envs):
            return self.action_space.shape
            
        def get_raw_action_shape(envs):
            return self.action_space.shape
        
        settings.include_info_keys.extend([
            ('transformed_action', get_transformed_action_shape),
            ('delta_transformation', get_delta_shape),
            ('raw_action', get_raw_action_shape),
        ])
        
        return settings 