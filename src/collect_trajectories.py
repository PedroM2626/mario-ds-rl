import os
import argparse
import pickle
import numpy as np
import shutil
from tqdm import tqdm
from sb3_contrib import RecurrentPPO
from env import MarioNdsEnv

def main():
    parser = argparse.ArgumentParser(description="Collect trajectories using a trained RecurrentPPO policy for Decision Transformer")
    parser.add_argument("--rom", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds", help="Path to NDS ROM")
    parser.add_argument("--state", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1", help="Path to Savestate")
    parser.add_argument("--model-path", type=str, default="models/recurrent_ppo_mario", help="Path to trained PPO model (.zip)")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of complete episodes to collect")
    parser.add_argument("--output-path", type=str, default="data/mario_trajectories.pkl", help="Output path for trajectories (.pkl)")
    args = parser.parse_args()

    # Create directory for output if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    temp_rom = "data/mario_temp.nds"
    temp_state = "data/state_temp.ds1"
    
    print("Creating temporary copies of ROM and Savestate to avoid file locking...")
    shutil.copyfile(args.rom, temp_rom)
    shutil.copyfile(args.state, temp_state)

    trajectories = []
    
    try:
        # Initialize environment
        print("Initializing Mario DS environment...")
        env = MarioNdsEnv(rom_path=temp_rom, state_path=temp_state)

        # Check if model exists
        model_file = args.model_path if args.model_path.endswith(".zip") else f"{args.model_path}.zip"
        if not os.path.exists(model_file):
            print(f"Warning: Trained model not found at {model_file}. Trajectories will be collected using a random policy.")
            model = None
        else:
            print(f"Loading trained policy on CPU from {model_file}...")
            # RecurrentPPO loads model, we pass env and device='cpu' to prevent VRAM allocation
            model = RecurrentPPO.load(model_file, env=env, device="cpu")
            print("Model loaded successfully!")

        for ep in range(args.num_episodes):
            print(f"Starting collection of Episode {ep + 1}/{args.num_episodes}...")
            
            obs, _ = env.reset()
            done = False
            truncated = False
            
            ep_obs = []
            ep_actions = []
            ep_rewards = []
            ep_dones = []
            
            # LSTM hidden states for RecurrentPPO
            lstm_states = None
            episode_start = np.array([True])
            
            step_count = 0
            
            while not (done or truncated):
                ep_obs.append(obs)
                
                if model is not None:
                    # SB3 RecurrentPPO expects batch format for observations
                    # Predict returns action and next lstm state.
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_start,
                        deterministic=True
                    )
                    action = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                else:
                    # Random action fallback bias right
                    if np.random.rand() < 0.6:
                        action = np.random.choice([1, 2, 3]) # Right, Right+B, Right+B+A
                    else:
                        action = env.action_space.sample()
                
                # Step the environment
                next_obs, reward, done, truncated, _ = env.step(action)
                
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_dones.append(done or truncated)
                
                obs = next_obs
                episode_start = np.array([done or truncated])
                step_count += 1
                
                if step_count % 100 == 0:
                    print(f"  Step {step_count}... Reward accumulated: {sum(ep_rewards):.1f}")
                    
            print(f"Episode {ep + 1} finished in {step_count} steps. Total reward: {sum(ep_rewards):.1f}")
            
            # Store trajectory
            trajectories.append({
                "observations": np.array(ep_obs, dtype=np.uint8),
                "actions": np.array(ep_actions, dtype=np.int64),
                "rewards": np.array(ep_rewards, dtype=np.float32),
                "dones": np.array(ep_dones, dtype=bool)
            })

    finally:
        print("Closing environment...")
        if 'env' in locals():
            env.close()
        
        print("Cleaning up temporary ROM and state copies...")
        if os.path.exists(temp_rom):
            try:
                os.remove(temp_rom)
            except Exception as e:
                print(f"Error removing temporary ROM: {e}")
        if os.path.exists(temp_state):
            try:
                os.remove(temp_state)
            except Exception as e:
                print(f"Error removing temporary state: {e}")

    print(f"Saving {len(trajectories)} trajectories to {args.output_path}...")
    with open(args.output_path, "wb") as f:
        pickle.dump(trajectories, f)
        
    print("Trajectory collection complete!")

if __name__ == "__main__":
    main()
