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

# Import the official SheepRL CLI evaluation runner
from sheeprl.cli import evaluation

if __name__ == "__main__":
    # Ensure checkpoint_path is provided
    if len(sys.argv) == 1:
        print("Error: You must provide a checkpoint_path.")
        print("Usage: python src/evaluate_dreamer.py checkpoint_path=path/to/checkpoint.ckpt")
        sys.exit(1)
        
    print(f"Starting SheepRL DreamerV3 evaluation with args: {sys.argv[1:]}")
    
    # Run the SheepRL CLI evaluation pipeline
    evaluation()
