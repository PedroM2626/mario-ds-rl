import os
import argparse
import numpy as np
from env import MarioNdsEnv
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Collect random gameplay frames for Unsupervised Learning")
    parser.add_argument("--rom", type=str, default="data/mario.nds", help="Path to NDS ROM")
    parser.add_argument("--state", type=str, default="data/state.dst", help="Path to Savestate")
    parser.add_argument("--num-frames", type=int, default=10000, help="Number of frames to collect")
    parser.add_argument("--output", type=str, default="data/mario_dataset.npz", help="Output .npz file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    env = MarioNdsEnv(rom_path=args.rom, state_path=args.state)
    
    print(f"Starting data collection: {args.num_frames} frames...")
    
    frames = []
    obs, _ = env.reset()
    
    # We loop and collect frames
    for i in tqdm(range(args.num_frames)):
        # Epsilon-greedy random action (bias towards moving right/jumping so it actually explores)
        # 0: Noop, 1: Right, 2: Right+B, 3: Right+B+A, 4: Left, 5: A
        if np.random.rand() < 0.6:
            action = np.random.choice([1, 2, 3]) # 60% chance to go right
        else:
            action = env.action_space.sample() # 40% purely random
            
        obs, _, done, truncated, _ = env.step(action)
        
        # obs is shape (84, 84, 1)
        frames.append(obs)
        
        if done or truncated:
            obs, _ = env.reset()
            
    env.close()
    
    print(f"Collection finished. Saving to {args.output}...")
    frames_array = np.array(frames, dtype=np.uint8)
    np.savez_compressed(args.output, images=frames_array)
    print(f"Dataset saved! Shape: {frames_array.shape}, Size: {frames_array.nbytes / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
