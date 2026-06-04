import os
import sys
import argparse
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch
from env import MarioNdsEnv
from impala_cnn import ImpalaCNNFeaturesExtractor

class MLflowCallback(BaseCallback):
    """
    Custom callback for logging to MLflow.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                mlflow.log_metric("episode_reward", info["episode"]["r"], step=self.num_timesteps)
                mlflow.log_metric("episode_length", info["episode"]["l"], step=self.num_timesteps)
        return True

def make_env(rom_path, state_path, rank):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = MarioNdsEnv(rom_path=rom_path, state_path=state_path)
        env = Monitor(env)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train Mario RL Agent with ImpalaCNN")
    parser.add_argument("--rom", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds", help="Path to NDS ROM")
    parser.add_argument("--state", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1", help="Path to Savestate")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing model to resume training (e.g. models/ppo_mario)")
    parser.add_argument("--num-envs", type=int, default=6, help="Number of parallel environments to run")
    parser.add_argument("--run-id", type=str, default="mario_impala", help="Name/ID for this training run")
    parser.add_argument("--n-steps", type=int, default=1376, help="Number of PPO steps per rollout per environment")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate for PPO training")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient for PPO")
    args = parser.parse_args()

    # Ensure paths are correct
    os.makedirs(os.path.dirname(args.rom), exist_ok=True)
    if not os.path.exists(args.rom):
        print(f"Warning: ROM not found at {args.rom}.")
    if not os.path.exists(args.state):
        print(f"Warning: Savestate not found at {args.state}.")

    # Initialize MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_NDS_RL")

    # MLOps Context
    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", "PPO_ImpalaCNN")
        mlflow.log_param("total_timesteps", args.timesteps)
        mlflow.log_param("frameskip", 8)
        
        # Parallel environments
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
        env_fns = [make_env(args.rom, args.state, i) for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)
        env = VecFrameStack(env, n_stack=4)

        # Policy Kwargs to use Custom ImpalaCNN Extractor
        policy_kwargs = dict(
            features_extractor_class=ImpalaCNNFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=512)
        )

        ent_coef_val = args.ent_coef
        n_steps_val = args.n_steps
        lr_val = args.lr

        if args.resume and os.path.exists(f"{args.resume}.zip"):
            print(f"Resuming training from {args.resume}.zip with ImpalaCNN...")
            model = PPO.load(args.resume, env=env, ent_coef=ent_coef_val, n_steps=n_steps_val, learning_rate=lr_val)
        else:
            print("Initializing PPO with ImpalaCNN policy...")
            model = PPO(
                "CnnPolicy", 
                env, 
                verbose=1, 
                ent_coef=ent_coef_val, 
                n_steps=n_steps_val, 
                learning_rate=lr_val, 
                tensorboard_log="./tensorboard_logs/", 
                policy_kwargs=policy_kwargs
            )

        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("ent_coef", ent_coef_val)
        mlflow.log_param("num_envs", args.num_envs)
        mlflow.log_param("n_steps", model.n_steps)
        mlflow.log_param("batch_size", model.batch_size)

        print("Starting training with ImpalaCNN...")
        
        # Save checkpoints every 10,000 steps of loop (60,000 timesteps total)
        save_freq = 10000
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path='./models/checkpoints/',
            name_prefix=args.run_id,
            save_vecnormalize=True
        )

        try:
            reset_ts = False if args.resume else True
            model.learn(
                total_timesteps=args.timesteps, 
                callback=[MLflowCallback(), checkpoint_callback], 
                reset_num_timesteps=reset_ts
            )
        except KeyboardInterrupt:
            print("\nTreinamento interrompido pelo usuário! Salvando progresso...")

        os.makedirs("models", exist_ok=True)
        model_path = f"models/{args.run_id}"
        model.save(model_path)
        print(f"Training complete. Model saved to {model_path}.")
        
        mlflow.log_artifact(f"{model_path}.zip", artifact_path="models")
        print("Model registered in MLflow.")

if __name__ == "__main__":
    main()
