#!/usr/bin/env python3

import sys
import subprocess
import os
from datetime import datetime

def get_user_config():
    """
    Get configuration from user input
    """
    print("\n" + "=" * 60)
    print("GROUNDING CONFIGURATION")
    print("=" * 60)
    
    print("\n1. 환경 설정:")
    src_env = input("Source Environment [HalfCheetah-v3]: ").strip() or "HalfCheetah-v3"
    trg_env = input("Target Environment [HalfCheetahBroken-v2]: ").strip() or "HalfCheetahBroken-v2"
    
    print("\n2. 파일 경로 설정:")
    print("전문가 정책 경로들을 입력하세요 (쉼표로 구분):")
    rollout_policy_path = input("Rollout Policy Path(s): ").strip()
    
    print("\n전문가 trajectory 파일 경로:")
    traj_load_path = input("Expert Trajectory Path: ").strip()
    
    print("\n3. 학습 설정:")
    training_steps_atp = input("ATP Training Steps [50000]: ").strip() or "50000"
    training_steps_policy = input("Policy Training Steps [50000]: ").strip() or "50000"
    n_transitions = input("Number of Expert Transitions [500]: ").strip() or "500"
    
    print("\n4. 실행 설정:")
    device = input("Device [cuda/cpu] [cuda]: ").strip() or "cuda"
    namespace = input("Experiment Namespace [interactive_grounding]: ").strip() or "interactive_grounding"
    
    return {
        'src_env': src_env,
        'trg_env': trg_env,
        'demo_sub_dir': 'BrokenCheetah' if 'Broken' in trg_env else 'HalfCheetah',
        'rollout_set': 'MS',
        'training_steps_atp': int(training_steps_atp),
        'training_steps_policy': int(training_steps_policy),
        'namespace': namespace,
        'expt_number': 1,
        'deterministic_atp': True,
        'verbose': 1,
        'n_transitions': int(n_transitions),
        'num_src': len(rollout_policy_path.split(',')) if rollout_policy_path else 1,
        'rollout_policy_path': rollout_policy_path if rollout_policy_path else None,
        'traj_load_path': traj_load_path if traj_load_path else None,
        'alg': 'diffATP',
        'env_name': 'augmented_MDP-v0',
        'device': device,
        'num_processes': 1,
        'deter_rollout': True,
        'collect_demo': False,
        'eval': True,
        'plot': True,
        'no_wb': True
    }

def get_default_config():
    """
    Get default test configuration
    """
    return {
        'src_env': 'HalfCheetah-v3',
        'trg_env': 'HalfCheetahBroken-v2', 
        'demo_sub_dir': 'BrokenCheetah',
        'rollout_set': 'MS',
        'training_steps_atp': 50000,
        'training_steps_policy': 50000,
        'namespace': 'default_grounding_test',
        'expt_number': 1,
        'deterministic_atp': True,
        'verbose': 1,
        'n_transitions': 500,
        'num_src': 1,
        'rollout_policy_path': None,  # Will trigger error handling in run_policy_grounding.py
        'traj_load_path': None,
        'alg': 'diffATP',
        'env_name': 'augmented_MDP-v0',
        'device': 'cuda',
        'num_processes': 1,
        'deter_rollout': True,
        'collect_demo': False,
        'eval': True,
        'plot': True,
        'no_wb': True
    }

def scan_for_existing_files():
    """
    Scan for existing policy and trajectory files
    """
    print("\n" + "=" * 60)
    print("기존 파일 검색 중...")
    print("=" * 60)
    
    # Search for policy files
    policy_patterns = [
        "data/models/initial_policies/**/*.pt",
        "expert_datasets/**/*.pt", 
        "**/*policy*.pt",
        "**/*expert*.pt"
    ]
    
    found_files = []
    import glob
    
    for pattern in policy_patterns:
        files = glob.glob(pattern, recursive=True)
        found_files.extend(files)
    
    if found_files:
        print("\n발견된 파일들:")
        for i, file_path in enumerate(found_files[:10], 1):  # Show first 10
            print(f"  {i}. {file_path}")
        if len(found_files) > 10:
            print(f"  ... 그리고 {len(found_files) - 10}개 더")
    else:
        print("\n❌ 정책/trajectory 파일을 찾을 수 없습니다.")
        print("다음 명령어로 파일을 검색해보세요:")
        print("  find . -name '*.pt' | head -10")
    
    return found_files

def run_grounding_test():
    """
    Run a grounding test with user configuration
    """
    print("=" * 60)
    print("GROUNDING TEST - Python Script")
    print(f"Start time: {datetime.now()}")
    print("=" * 60)
    
    # Get configuration from user
    config = get_user_config()
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 60)
    
    # Build command
    cmd = [sys.executable, 'diffatp/run_policy_grounding.py']  # Use current Python interpreter
    
    for key, value in config.items():
        if value is not None:  # Skip None values
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])  # Convert underscores to hyphens
    
    print("Command to execute:")
    print(" ".join(cmd))
    print("-" * 60)
    
    # Execute command
    try:
        print("Starting grounding experiment...")
        env = os.environ.copy()  # Copy current environment (including conda env)
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.getcwd(), env=env)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ GROUNDING TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ GROUNDING TEST FAILED!")
            print(f"Return code: {result.returncode}")
            print("=" * 60)
            return False
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("⚠️  GROUNDING TEST INTERRUPTED BY USER")
        print("=" * 60)
        return False
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ GROUNDING TEST ERROR: {e}")
        print("=" * 60)
        return False

def run_quick_test():
    """
    Run a very quick test with minimal steps
    """
    print("=" * 60)
    print("QUICK GROUNDING TEST - Minimal Configuration")
    print("=" * 60)
    
    cmd = [
        sys.executable, 'diffatp/run_policy_grounding.py',  # Use current Python interpreter
        '--src_env', 'HalfCheetah-v3',
        '--trg_env', 'HalfCheetah-v3',  # Same env for testing
        '--demo_sub_dir', 'HalfCheetah',
        '--rollout_set', 'MS', 
        '--training_steps_atp', '1000',  # Very small for quick test
        '--training_steps_policy', '1000',  # Very small for quick test
        '--namespace', 'quick_test',
        '--expt_number', '1',
        '--deterministic_atp', 'True',
        '--verbose', '1',
        '--n-transitions', '100',  # Very small dataset
        '--num_src', '1',
        '--alg', 'diffATP',
        '--env_name', 'augmented_MDP-v0',
        '--device', 'cpu',  # Use CPU for quick test
        '--no_wb', 'True'  # No wandb logging
    ]
    
    print("Quick test command:")
    print(" ".join(cmd))
    print("-" * 60)
    
    try:
        env = os.environ.copy()  # Copy current environment (including conda env)
        result = subprocess.run(cmd, cwd=os.getcwd(), env=env)
        return result.returncode == 0
    except Exception as e:
        print(f"Quick test error: {e}")
        return False

def main():
    """
    Main test function
    """
    print("Python Grounding Test Script")
    print("Choose test type:")
    print("1. 인터랙티브 설정 (경로 직접 입력)")
    print("2. 파일 검색 후 설정") 
    print("3. 기본 설정으로 빠른 테스트")
    print("4. 최소 설정 테스트 (CPU)")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1/2/3/4/5): ").strip()
            
            if choice == '1':
                print("\n인터랙티브 설정 모드...")
                success = run_grounding_test()
                break
                
            elif choice == '2':
                print("\n파일 검색 후 설정...")
                found_files = scan_for_existing_files()
                if found_files:
                    print("\n발견된 파일들을 참고하여 경로를 입력하세요.")
                    success = run_grounding_test()
                else:
                    print("\n파일을 찾을 수 없으므로 기본 설정으로 진행합니다.")
                    success = run_default_test()
                break
                
            elif choice == '3':
                print("\n기본 설정으로 테스트...")
                success = run_default_test()
                break
                
            elif choice == '4':
                print("\n최소 설정 테스트...")
                success = run_quick_test()
                break
                
            elif choice == '5':
                print("Exiting...")
                return
                
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
                continue
                
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    print(f"\nTest completed. Success: {success}")

def run_default_test():
    """
    Run test with default configuration
    """
    print("=" * 60)
    print("DEFAULT GROUNDING TEST")
    print("=" * 60)
    
    config = get_default_config()
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 60)
    
    # Build command
    cmd = [sys.executable, 'diffatp/run_policy_grounding.py']
    
    for key, value in config.items():
        if value is not None:  # Skip None values
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])  # Convert underscores to hyphens
    
    print("Command to execute:")
    print(" ".join(cmd))
    print("-" * 60)
    
    try:
        print("Starting default grounding test...")
        env = os.environ.copy()
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.getcwd(), env=env)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ DEFAULT TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ DEFAULT TEST FAILED!")
            print(f"Return code: {result.returncode}")
            print("=" * 60)
            return False
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("⚠️  TEST INTERRUPTED BY USER")
        print("=" * 60)
        return False
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST ERROR: {e}")
        print("=" * 60)
        return False

if __name__ == "__main__":
    # Check if run_policy_grounding.py exists
    if not os.path.exists('diffatp/run_policy_grounding.py'):
        print("Error: diffatp/run_policy_grounding.py not found!")
        print("Make sure you're running this script from the project root directory.")
        sys.exit(1)
    
    main() 