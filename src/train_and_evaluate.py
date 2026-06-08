import os
import sys
import subprocess
import glob

def get_latest_checkpoint(base_dir):
    # Find all checkpoint files under logs/runs/dreamer_v3/MarioNDS-Dreamer-v0/
    pattern = os.path.join(base_dir, "logs/runs/dreamer_v3/MarioNDS-Dreamer-v0/**/checkpoint/*.ckpt")
    ckpt_files = glob.glob(pattern, recursive=True)
    if not ckpt_files:
        return None
    # Return the one with the newest modification time
    return max(ckpt_files, key=os.path.getmtime)

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Run training
    print("=== Starting DreamerV3 500k Steps Training ===")
    train_cmd = [sys.executable, "src/train_dreamer.py"]
    
    # Forward any arguments passed to this script to the training script
    if len(sys.argv) > 1:
        train_cmd.extend(sys.argv[1:])
        
    result = subprocess.run(train_cmd, cwd=project_dir)
    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        sys.exit(result.returncode)
        
    print("=== Training Completed Successfully ===")
    
    # 2. Find latest checkpoint
    latest_ckpt = get_latest_checkpoint(project_dir)
    if not latest_ckpt:
        print("Error: No checkpoint file found after training!")
        sys.exit(1)
        
    print(f"Found latest checkpoint: {latest_ckpt}")
    
    # Convert path to relative to project directory
    rel_ckpt = os.path.relpath(latest_ckpt, project_dir)
    
    # 3. Run evaluation in human render mode
    print("=== Starting Evaluation / Visualization (stochastic) ===")
    env = os.environ.copy()
    env["MARIO_RENDER_MODE"] = "human"
    
    eval_cmd = [
        sys.executable, 
        "src/evaluate_dreamer.py", 
        f"checkpoint_path={rel_ckpt}", 
        "fabric.accelerator=cuda", 
        "env.capture_video=False"
    ]
    
    result = subprocess.run(eval_cmd, cwd=project_dir, env=env)
    if result.returncode != 0:
        print(f"Evaluation failed with return code {result.returncode}")
        sys.exit(result.returncode)
        
    print("=== Pipeline Finished Successfully ===")

if __name__ == "__main__":
    main()
