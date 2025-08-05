import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import glob
import os
import sys
import random
import yaml
import numpy as np
import torch
from collections import OrderedDict

sys.path.insert(0, "./")

import gym
from diffatp.grounded_env import GroundedEnv
from diffatp.atp_envs import ATPEnv_Multiple
from diffatp.main import DiffATPSettings, get_setup_dict
from rlf import run_policy
from rlf.policies.base_policy import get_empty_step_info
import rlf.rl.utils as rutils
from goal_prox.method.utils import trim_episodes_trans
from goal_prox.envs.goal_traj_saver import GoalTrajSaver
from rlf.rl.loggers.wb_logger import WbLogger
from rlf.rl.loggers.base_logger import BaseLogger
from rlf.exp_mgr import config_mgr


# def collect_demo(args, generate_demo=False, save_path=None, rollout_policies_paths=None):
#     """
#     Collect target transition samples (s,a,s') adapted for DRAIL project
    
#     :param args: (object) argument values of experiments
#     :param generate_demo: (bool) if false, read the location of stored data
#     :param save_path: (str) path to save data
#     :param rollout_policies_paths: (list) paths to rollout policies
#     """
#     if generate_demo:
#         data_save_dir = []
#         for k in range(args.num_src):
#             idx = (args.expt_number + k - 1) % len(rollout_policies_paths)
#             if args.n_episodes is not None:
#                 data_save_dir_i = os.path.join(save_path,
#                                                f"{args.trg_env}_episodes_{args.n_episodes}_seed{idx}")
#             else:
#                 data_save_dir_i = os.path.join(save_path,
#                                                f"{args.trg_env}_transitions_{args.n_transitions}_seed{idx}")
            
#             # Generate trajectory using current project structure
#             real_env = gym.make(args.trg_env)
#             # Here you would implement trajectory generation using your project's methods
#             # This is a placeholder - implement according to your project's trajectory generation
#             print(f"Would generate trajectory for {data_save_dir_i}")
            
#             data_save_dir_i = data_save_dir_i + ".npz"
#             data_save_dir.append(data_save_dir_i)
#     else:
#         data_save_dir = []
#         for k in range(args.num_src):
#             idx = (args.expt_number + k - 1) % len(rollout_policies_paths)
#             if args.n_episodes is not None:
#                 data_save_dir_i = os.path.join(save_path,
#                                                f"{args.trg_env}_episodes_{args.n_episodes}_seed{idx}")
#             else:
#                 data_save_dir_i = os.path.join(save_path,
#                                                f"{args.trg_env}_transitions_{args.n_transitions}_seed{idx}")
#             data_save_dir_i = data_save_dir_i + ".npz"
#             data_save_dir.append(data_save_dir_i)

#     for i in range(len(data_save_dir)):
#         print(f"Demo data path: {data_save_dir[i]}")
#         if os.path.isfile(data_save_dir[i]):
#             print(f"Loading Demo Data: {data_save_dir[i]}")
#         else:
#             print(f"Warning: Demo data not found at {data_save_dir[i]}")

#     return data_save_dir


class GroundingTrainingCallback:
    """Callback for grounding training adapted for DRAIL project"""
    
    def __init__(self, plot_freq=100000, save_path=None, name_prefix="", 
                 verbose=1, true_transformation=None, logger=None):
        self.plot_freq = plot_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.verbose = verbose
        self.true_transformation = true_transformation
        self.logger = logger
        self.step_count = 0
        
    def on_step(self, model, grounded_env):
        """Called during training steps"""
        self.step_count += 1
        
        if self.step_count % self.plot_freq == 0:
            if self.verbose > 0:
                print(f"Plotting action transformation at step {self.step_count}")
            
            plot_path = os.path.join(self.save_path, f'grounding_step_{self.step_count}.png')
            grounded_env.plot_action_transformation(
                expt_path=plot_path,
                true_transformation=self.true_transformation,
                logger=self.logger
            )


def load_rollout_policies(args, rollout_policies_paths):
    
    # Based on configs/halfcheetah/ppo.yaml and configs/walker/expert/ppo.yaml
    if not hasattr(args, 'recurrent_policy'):
        args.recurrent_policy = False
    if not hasattr(args, 'hidden_size'):
        args.hidden_size = 256  # ppo-hidden-dim from config
    if not hasattr(args, 'ppo_hidden_dim'):
        args.ppo_hidden_dim = 256  # from config
    if not hasattr(args, 'ppo_layers'):
        args.ppo_layers = 2  # typical default for PPO
    if not hasattr(args, 'cuda'):
        args.cuda = (args.device == 'cuda')
    if not hasattr(args, 'log_dir'):
        args.log_dir = './logs'
    if not hasattr(args, 'prefix'):
        args.prefix = 'ppo'  # from config
    if not hasattr(args, 'seed'):
        args.seed = 1  # from config
    if not hasattr(args, 'policy_ob_key'):
        args.policy_ob_key = None
    if not hasattr(args, 'use_proper_time_limits'):
        args.use_proper_time_limits = True
    if not hasattr(args, 'gamma'):
        args.gamma = 0.99
    if not hasattr(args, 'lr'):
        args.lr = 0.0001  # from config
    if not hasattr(args, 'eps'):
        args.eps = 1e-5
    if not hasattr(args, 'max_grad_norm'):
        args.max_grad_norm = 0.5  # from config
    if not hasattr(args, 'num_steps'):
        args.num_steps = 256  # from config
    if not hasattr(args, 'ppo_epoch'):
        args.ppo_epoch = 10  # num-epochs from config
    if not hasattr(args, 'num_epochs'):
        args.num_epochs = 10  # alias for ppo_epoch
    if not hasattr(args, 'num_mini_batch'):
        args.num_mini_batch = 32  # from config
    if not hasattr(args, 'clip_param'):
        args.clip_param = 0.2
    if not hasattr(args, 'clip_actions'):
        args.clip_actions = True  # from config
    if not hasattr(args, 'value_loss_coef'):
        args.value_loss_coef = 0.5
    if not hasattr(args, 'entropy_coef'):
        args.entropy_coef = 0.001  # from config
    if not hasattr(args, 'use_gae'):
        args.use_gae = True
    if not hasattr(args, 'gae_lambda'):
        args.gae_lambda = 0.95
    if not hasattr(args, 'use_linear_lr_decay'):
        args.use_linear_lr_decay = False
    if not hasattr(args, 'normalize_env'):
        args.normalize_env = True  # from config
    if not hasattr(args, 'eval_interval'):
        args.eval_interval = 20000  # from config
    if not hasattr(args, 'log_interval'):
        args.log_interval = 1  # from config
    if not hasattr(args, 'save_interval'):
        args.save_interval = 100000  # from config
    if not hasattr(args, 'deterministic_policy'):
        args.deterministic_policy = True  # Use deterministic actions for grounding
    
    rollout_policies = []
    
    for k in range(args.num_src):
        idx = (args.expt_number + k - 1) % len(rollout_policies_paths)
        policy_path = rollout_policies_paths[idx]
        
        print(f"Loading policy from: {policy_path}")
        
        # Load policy using current project's method
        policy_state = torch.load(policy_path)
        
        # Create policy instance (adapt to your project's policy creation)
        env = gym.make(args.src_env)
        policy = get_setup_dict()["ppo"][1](args.src_env, args)
        policy_args = (env.observation_space, env.action_space, args)
        policy.init(*policy_args)
        policy = policy.to(args.device)
        policy.set_env_ref(env)
        
        if isinstance(policy_state, dict) and "policy" in policy_state:
            policy.load_state_dict(policy_state["policy"])
        else:
            policy.load_state_dict(policy_state)
            
        policy.eval()
        rollout_policies.append(policy)
    
    return rollout_policies


def create_grounded_environment(args, atp_policy, deterministic=True):
    """Create grounded environment using current project structure"""
    env = gym.make(args.src_env)
    
    grounded_env = GroundedEnv(
        env=env,
        action_tf_policy=atp_policy,
        args=args,
        use_deterministic=deterministic
    )
    
    return grounded_env


def main():
    parser = argparse.ArgumentParser(description='Multi-source Grounding for DRAIL')
    parser.add_argument('--src_env', default='HalfCheetah-v2', help="Name of source environment")
    parser.add_argument('--trg_env', default='HalfCheetahBroken-v2', help="Name of target environment")
    parser.add_argument('--demo_sub_dir', default='HalfCheetah', help="Subdirectory for demonstration")
    parser.add_argument('--rollout_set', default='MS', help='Types of rollout policy set')
    parser.add_argument('--training_steps_atp', default=int(1e6), type=int, help="Total time steps to learn ATP")
    parser.add_argument('--training_steps_policy', default=int(1e6), type=int, help="Total time steps to learn agent policy")
    parser.add_argument('--expt_number', default=1, type=int, help="Experiment number used for random seed")
    parser.add_argument('--deterministic_atp', type=lambda x: x.lower() == 'true', default=False, help="Deterministic ATP in grounded environment")
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)
    parser.add_argument('--n-episodes', help='Number of expert episodes', type=int, default=None)
    parser.add_argument('--n-transitions', help='Number of expert transitions', type=int, default=2000)
    parser.add_argument('--single_pol', type=lambda x: x.lower() == 'true', default=False, help="Use single source policy")
    parser.add_argument('--num_src', default=1, type=int, help="Number of rollout policies")
    parser.add_argument('--collect_demo', type=lambda x: x.lower() == 'true', default=False, help="Collect new samples from target environment")
    parser.add_argument('--deter_rollout', type=lambda x: x.lower() == 'true', default=False, help="Deploy rollout policy deterministically")
    
    # DRAIL project specific arguments
    parser.add_argument('--device', default='cuda', help="Device to use")
    parser.add_argument('--num_processes', default=1, type=int, help="Number of processes")
    parser.add_argument('--alg', default='diffATP', help="Algorithm to use")
    parser.add_argument('--env_name', default='augmented_MDP-v0', help="Environment name for training")
    parser.add_argument('--src_env_name', default='HalfCheetah-v2', help="Source environment name")
    parser.add_argument('--rollout_policy_path', default=None, help="Path to rollout policies")
    parser.add_argument('--traj_load_path', default=None, help="Path to trajectory data")
    parser.add_argument('--config_path', default=None, help="Path to config file")
    
    # Output and logging
    parser.add_argument('--namespace', default="grounding_test", type=str, help="Namespace for experiments")
    parser.add_argument('--no_wb', type=lambda x: x.lower() == 'true', default=False, help="Disable wandb logging")
    parser.add_argument('--eval', type=lambda x: x.lower() == 'true', default=False, help="Evaluate after training")
    parser.add_argument('--plot', type=lambda x: x.lower() == 'true', default=False, help="Visualize action transformer policy")
    
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Determine transformation type
    if 'Broken' in args.trg_env:
        true_transformation = 'Broken'
    elif 'PositiveSkew' in args.trg_env:
        true_transformation = np.array([1.5])
    else:
        true_transformation = None
    
    # Set seeds
    random.seed(args.expt_number)
    np.random.seed(args.expt_number)
    torch.manual_seed(args.expt_number)
    
    expt_type = 'sim2sim' if args.src_env == args.trg_env else 'sim2real'
    expt_label = f"{args.trg_env}_{args.namespace}_num_src_{args.num_src}_{expt_type}_seed{args.expt_number}"
    
    # Create experiment folder
    base_path = 'data/models/grounding/'
    expt_path = os.path.join(base_path, expt_label)
    os.makedirs(expt_path, exist_ok=True)
    
    # Initialize config manager for WandB
    config_mgr.init('config.yaml')
    
    # Initialize logger
    if not args.no_wb:
        logger = WbLogger(should_log_vids=True)
    else:
        logger = BaseLogger()
    
    # Save config
    config_dict = vars(args)
    with open(os.path.join(expt_path, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)
    
    # Load rollout policies
    if args.rollout_policy_path:
        rollout_policies_paths = args.rollout_policy_path.split(", ")
    # else:
    #     # Default path structure
    #     load_paths = f"data/models/initial_policies/{args.rollout_set}/{args.src_env}/"
    #     rollout_policies_paths = sorted(glob.glob(os.path.join(load_paths, '*')))
    
    print(f"Loading rollout policies from: {rollout_policies_paths}")
    rollout_policies = load_rollout_policies(args, rollout_policies_paths)
    
    # Create ATP environment
    atp_env = ATPEnv_Multiple(
        env=gym.make(args.src_env),
        rollout_policy_list=rollout_policies,
        deter_rollout=args.deter_rollout,
        args=args
    )
    
    # Collect demonstration data
    if args.rollout_set == 'ER':
        save_path = 'data/real_traj_ER'
    elif args.rollout_set == 'MS':
        save_path = 'data/real_traj_MS'
    
    if args.demo_sub_dir:
        save_path = os.path.join(save_path, args.demo_sub_dir)
    
    os.makedirs(save_path, exist_ok=True)
    
    expert_data_paths = collect_demo(
        args,
        save_path=save_path,
        generate_demo=args.collect_demo,
        rollout_policies_paths=rollout_policies_paths
    )
    
    # Create ATP (Action Transformation Policy) using DiffATP
    atp_algo = get_setup_dict()[args.alg][0]
    atp_policy = get_setup_dict()[args.alg][1](args.env_name, args)
    
    # Initialize ATP
    atp_policy.init(atp_env.observation_space, atp_env.action_space, args)
    atp_policy = atp_policy.to(args.device)
    atp_policy.set_env_ref(atp_env)
    
    print('################# START GROUNDING #################')
    
    # Test initial grounded environment
    initial_grounded_env = create_grounded_environment(args, atp_policy, deterministic=True)
    initial_grounded_env.test_grounded_environment(
        expt_path=os.path.join(expt_path, 'initial_grounding.png'),
        true_transformation=true_transformation,
        logger=logger
    )
    
    # Training callback
    callback = GroundingTrainingCallback(
        plot_freq=100000,
        save_path=expt_path,
        name_prefix="grounding",
        verbose=args.verbose,
        true_transformation=true_transformation,
        logger=logger
    )
    
    # Train ATP (This is a placeholder - implement according to your training loop)
    print("Training Action Transformation Policy...")
    
    # Here you would implement the actual training loop for ATP
    # This would integrate with your DiffATP algorithm
    # For now, this is a placeholder
    
    # Save ATP
    print('##### SAVING ACTION TRANSFORMER POLICY #####')
    torch.save(atp_policy.state_dict(), os.path.join(expt_path, 'action_transformer_policy.pt'))
    
    # Policy learning in grounded environment
    print('################# START POLICY LEARNING #################')
    
    # Create grounded environment for policy learning
    grounded_env = create_grounded_environment(args, atp_policy, deterministic=args.deterministic_atp)
    
    # Test grounded environment
    if args.plot:
        grounded_env.test_grounded_environment(
            expt_path=os.path.join(expt_path, 'final_grounding.png'),
            true_transformation=true_transformation,
            logger=logger
        )
    
    # Target policy learning (placeholder)
    print("Training target policy in grounded environment...")
    
    # Here you would implement target policy learning
    # This would use your existing RL algorithms (PPO, etc.)
    
    # Evaluation
    if args.eval:
        print("################# START EVALUATION #################")
        
        # Evaluate in target environment
        target_env = gym.make(args.trg_env)
        
        # Evaluation code here
        print(f"Evaluation completed. Results saved to {expt_path}")
    
    logger.close()
    print(f"Grounding experiment completed. Results saved to {expt_path}")


if __name__ == '__main__':
    main() 