import argparse
import time
from stable_baselines3 import PPO
from env import MarioNdsEnv

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mario RL Agent")
    parser.add_argument("--model", type=str, default="models/ppo_mario.zip", help="Path to trained model")
    parser.add_argument("--rom", type=str, default="../data/mario.nds", help="Path to NDS ROM")
    parser.add_argument("--state", type=str, default="../data/state.dst", help="Path to Savestate")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    args = parser.parse_args()

    env = MarioNdsEnv(rom_path=args.rom, state_path=args.state, render_mode="human")
    
    try:
        model = PPO.load(args.model, env=env)
        print(f"Successfully loaded model from {args.model}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.01) # Slight pause to make it watchable
            
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
        
    env.close()

if __name__ == "__main__":
    main()
