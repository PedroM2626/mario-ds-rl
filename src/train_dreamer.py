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
    defaults = {
        "exp": "dreamer_v3",
        "env": "gym",
        "env.id": "MarioNDS-Dreamer-v0",
        "env.num_envs": "1",
        "fabric.accelerator": "cuda",
        "fabric.devices": "1",
        "algo.total_steps": "500000",
        "checkpoint.every": "1000",
        "buffer.size": "100000",
        "buffer.memmap": "false",
        "buffer.checkpoint": "false",
        "env.capture_video": "false",
    }
    
    # Parse existing arguments
    args_dict = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            args_dict[k] = v
        else:
            args_dict[arg] = None
            
    # Merge defaults
    for k, v in defaults.items():
        if k not in args_dict:
            args_dict[k] = v
            
    # Reconstruct sys.argv
    new_args = [sys.argv[0]]
    for k, v in args_dict.items():
        if v is not None:
            new_args.append(f"{k}={v}")
        else:
            new_args.append(k)
            
    sys.argv = new_args
        
    print(f"Starting SheepRL DreamerV3 training with args: {sys.argv[1:]}")
    
    # Run the SheepRL CLI pipeline (Hydra parser + Fabric strategy distributor + agent optimization loop)
    run()
