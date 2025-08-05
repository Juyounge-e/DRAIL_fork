import subprocess
import sys
import os

def main():
    print("=" * 60)
    print(" Policy Grounding Test Script")
    print("=" * 60)
    
    
    # 1. 환경 설정
    src_env = "HalfCheetah-v2"  # 소스 환경
    trg_env = "HalfCheetahBroken-v2"  # 타겟 환경
    
    # 2. 파일 경로 설정 (여기를 본인 경로로 수정하세요!)
    rollout_policy_path = "./data/trained_models/HalfCheetah-v2/73-HC-1-PJ-ppo/model_3051.pt"  # 롤아웃 정책 경로
    traj_load_path = "./data/trained_models/HalfCheetahBroken-v2/75-HCB-1-6T-ppo/model_3051.pt"     # 전문가 trajectory 경로
    
    # 3. 학습 설정
    training_steps_atp = 1000      # ATP 학습 스텝
    training_steps_policy = 1000   # 정책 학습 스텝
    n_transitions = 100            # 전문가 transition 수
    
    # 4. 실행 설정
    device = "cuda"                    # cuda 또는 cpu
    namespace = "test_grounding"       # 실험 네임스페이스
    expt_number = 1                    # 실험 번호
    no_wb = True                       # WandB 사용 안함 (True로 설정 권장)
    
    # Configuration dictionary
    config = {
        'src_env': src_env,
        'trg_env': trg_env,
        'demo_sub_dir': 'BrokenCheetah',
        'rollout_set': 'MS',
        'training_steps_atp': training_steps_atp,
        'training_steps_policy': training_steps_policy,
        'namespace': namespace,
        'expt_number': expt_number,
        'deterministic_atp': True,
        'verbose': 1,
        'n-transitions': n_transitions,
        'num_src': 1,
        'rollout_policy_path': rollout_policy_path,
        'traj_load_path': traj_load_path,
        'alg': 'diffATP',
        'env_name': 'augmented_MDP-v0',
        'device': device,
        'num_processes': 1,
        'deter_rollout': True,
        'collect_demo': False,
        'eval': True,
        'plot': True,
        'no_wb': no_wb
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("-" * 60)
    
    # Build command
    cmd = [sys.executable, 'diffatp/run_policy_grounding.py']
    
    for key, value in config.items():
        if value is not None:  # Skip None values
            cmd.extend([f'--{key}', str(value)])  # Keep underscores as-is
    
    print("Command to execute:")
    print(' '.join(cmd))
    print("-" * 60)
    
    # Execute command
    print("Starting grounding experiment...")
    try:
        # Set environment variables for subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd() + ':' + env.get('PYTHONPATH', '')
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,  # 실시간 출력을 위해 False로 설정
            text=True
        )
        
        print("=" * 60)
        if result.returncode == 0:
            print(" GROUNDING TEST COMPLETED SUCCESSFULLY!")
        else:
            print(" GROUNDING TEST FAILED!")
            print(f"Return code: {result.returncode}")
        print("=" * 60)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f" ERROR: {e}")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nTest completed. Success: {success}") 