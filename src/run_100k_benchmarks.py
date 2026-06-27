import subprocess
import sys
import time

def run_command(command):
    print(f"\n==================================================")
    print(f"Running benchmark command:\n{' '.join(command)}")
    print(f"==================================================")
    
    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    # Print stdout in real-time
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    elapsed = time.time() - start_time
    
    if process.returncode == 0:
        print(f"\nCommand completed successfully in {elapsed:.1f} seconds.")
        return True
    else:
        print(f"\nCommand failed with exit code {process.returncode} after {elapsed:.1f} seconds.")
        return False

def main():
    print("Starting 100k Steps Benchmark Suite for Mario DS RL...")
    print("These runs will execute sequentially to prevent GPU Out Of Memory (OOM) errors.")
    
    benchmarks = [
        # 1. PPO Recurrent + DrQ (Data Regularized Q-learning style random shifts)
        [
            sys.executable, "src/train_ppo_recurrent.py",
            "--timesteps", "100000",
            "--num-envs", "4",
            "--use-drq",
            "--run-id", "ppo_drq_100k"
        ],
        # 2. PPO Recurrent + SPR (Self-Supervised Policy Representations)
        [
            sys.executable, "src/train_ppo_recurrent.py",
            "--timesteps", "100000",
            "--num-envs", "4",
            "--use-spr",
            "--run-id", "ppo_spr_100k"
        ],
        # 3. PPO Causal Transformer + SPR (Substituição da LSTM por Transformer + SPR)
        [
            sys.executable, "src/train_ppo_recurrent.py",
            "--timesteps", "100000",
            "--num-envs", "4",
            "--policy-type", "transformer",
            "--use-spr",
            "--run-id", "ppo_transformer_spr_100k"
        ]
    ]
    
    success_count = 0
    for i, cmd in enumerate(benchmarks):
        print(f"\nProgress: Run {i+1}/{len(benchmarks)}")
        success = run_command(cmd)
        if success:
            success_count += 1
        else:
            print(f"Benchmark run {i+1} encountered an error. Continuing to next run...")
            
    print(f"\nAll benchmark runs completed! Success rate: {success_count}/{len(benchmarks)}")

if __name__ == "__main__":
    main()
