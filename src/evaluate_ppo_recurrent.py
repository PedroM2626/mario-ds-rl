import argparse
import time
import sys
import os
import numpy as np

# Ensure src is in path for custom imports
sys.path.append("src")

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import MarioNdsEnv

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mario RL Agent with Recurrent PPO")
    parser.add_argument("--model", type=str, default="models/checkpoints/recurrent_ppo_mario_390000_steps.zip", help="Path to trained model")
    parser.add_argument("--rom", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds", help="Path to NDS ROM")
    parser.add_argument("--state", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1", help="Path to Savestate")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--stochastic", action="store_true", help="Evaluate stochastically")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return

    # Initialize environment
    # Override MARIO_RENDER_MODE to human so the window pops up
    os.environ["MARIO_RENDER_MODE"] = "human"
    
    # Recurrent PPO uses DummyVecEnv but NO FrameStack
    env = MarioNdsEnv(rom_path=args.rom, state_path=args.state, render_mode="human")
    env = DummyVecEnv([lambda: env])

    print(f"Loading Recurrent PPO model from {args.model}...")
    try:
        model = RecurrentPPO.load(args.model, env=env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    for episode in range(args.episodes):
        obs = env.reset()
        # Recurrent policies require hidden states
        lstm_states = None
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((1,), dtype=bool)
        
        done = [False]
        total_reward = 0
        
        print(f"\n--- Starting Episode {episode + 1} ---")
        while not done[0]:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=not args.stochastic,
            )
            obs, reward, done, info = env.step(action)
            episode_starts = done
            total_reward += reward[0]
            
            # The environment already calls cv2.imshow when render_mode="human"
            time.sleep(0.01) # Slight pause to make it watchable
            
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
        
    env.close()

if __name__ == "__main__":
    main()
