import argparse
import time
from stable_baselines3 import PPO
from env import MarioNdsEnv

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mario RL Agent")
    parser.add_argument("--model", type=str, default="models/ppo_mario.zip", help="Path to trained model")
    parser.add_argument("--rom", type=str, default="data/mario.nds", help="Path to NDS ROM")
    parser.add_argument("--state", type=str, default="data/state.dst", help="Path to Savestate")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--stochastic", action="store_true", help="Evaluate stochastically (useful to see untrained models mash buttons)")
    parser.add_argument("--use-autoencoder", action="store_true", help="Use pre-trained Autoencoder for vision")
    parser.add_argument("--use-icm", action="store_true", help="Use Intrinsic Curiosity Module (ICM)")
    args = parser.parse_args()
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    
    env = MarioNdsEnv(rom_path=args.rom, state_path=args.state, render_mode="human")
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    
    if args.use_icm:
        from icm import ICMVecEnvWrapper
        env = ICMVecEnvWrapper(env)
        
    if args.use_autoencoder:
        from train import CustomAutoencoderFeaturesExtractor
    
    try:
        model = PPO.load(args.model, env=env)
        print(f"Successfully loaded model from {args.model}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    for episode in range(args.episodes):
        obs = env.reset()
        done = [False]
        total_reward = 0
        
        while not done[0]:
            action, _states = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            env.render()
            time.sleep(0.01) # Slight pause to make it watchable
            
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
        
    env.close()

if __name__ == "__main__":
    main()
