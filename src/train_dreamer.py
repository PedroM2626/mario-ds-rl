import sys
import os

# Append the directory containing this script to sys.path
# to ensure python can import local modules
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import our custom environment to ensure it gets registered with gymnasium
import sheeprl_env

# Import the official SheepRL CLI runner
from sheeprl.cli import run

if __name__ == "__main__":
    # If the script is run without arguments, set the default parameters for DreamerV3 training
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "exp=dreamer_v3",
            "env=gym",
            "env.id=MarioNDS-Dreamer-v0",
            "env.num_envs=1",
            "fabric.accelerator=cuda",
            "fabric.devices=1",
            "algo.total_steps=500000",
            "checkpoint.every=50000",
            # Optimize replay buffer for machines with limited disk space
            # DreamerV3 defaults to 1,000,000 steps which takes ~7GB of disk memmap space
            "buffer.size=100000",
            "buffer.memmap=false", # Keep buffer in RAM to avoid disk full errors (Errno 28)
        ]
        
    print(f"Starting SheepRL DreamerV3 training with args: {sys.argv[1:]}")
    
    # Run the SheepRL CLI pipeline (Hydra parser + Fabric strategy distributor + agent optimization loop)
    run()
