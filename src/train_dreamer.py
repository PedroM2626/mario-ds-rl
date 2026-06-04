import sys
import os
import torch

# Monkey-patch torch.load to default weights_only=False for PyTorch 2.6 compatibility.
# In PyTorch 2.6, weights_only defaults to True, which raises exceptions when loading
# custom serialized classes like SheepRL's ReplayBuffer from checkpoints.
original_load = torch.load
def custom_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = custom_load
print("Applied PyTorch 2.6 compatibility patch (weights_only=False by default)")

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
            # Save checkpoints every 1000 steps (~15 minutes of training) as requested by user
            "checkpoint.every=1000",
            # Optimize replay buffer for machines with limited disk space
            # DreamerV3 defaults to 1,000,000 steps which takes ~7GB of disk memmap space
            "buffer.size=100000",
            "buffer.memmap=false", # Keep buffer in RAM to avoid disk full errors (Errno 28)
        ]
        
    print(f"Starting SheepRL DreamerV3 training with args: {sys.argv[1:]}")
    
    # Run the SheepRL CLI pipeline (Hydra parser + Fabric strategy distributor + agent optimization loop)
    run()
