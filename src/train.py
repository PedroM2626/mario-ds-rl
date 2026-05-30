import os
import argparse
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from env import MarioNdsEnv

class MLflowCallback(BaseCallback):
    """
    Custom callback for logging to MLflow.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # If the environment is vectorized, check infos for episode data
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                
                # Log to MLflow
                mlflow.log_metric("episode_reward", info["episode"]["r"], step=self.num_timesteps)
                mlflow.log_metric("episode_length", info["episode"]["l"], step=self.num_timesteps)
        return True

class CustomAutoencoderFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512, model_path="models/autoencoder.pth"):
        super().__init__(observation_space, features_dim)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, features_dim),
            nn.ReLU()
        )
        
        if os.path.exists(model_path):
            print(f"Loading pre-trained Autoencoder weights from {model_path}...")
            state_dict = torch.load(model_path, map_location="cpu")
            encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            
            # Adapt 1-channel pre-trained weights to 4-channel FrameStack
            if encoder_state_dict['0.weight'].shape[1] == 1 and observation_space.shape[0] == 4:
                w = encoder_state_dict['0.weight']
                encoder_state_dict['0.weight'] = w.repeat(1, 4, 1, 1) / 4.0
                
            self.encoder.load_state_dict(encoder_state_dict)
            
            # Freeze the convolutional layers
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Autoencoder weights loaded and frozen!")
        else:
            print("Warning: Autoencoder weights not found. Using randomly initialized encoder.")

    def forward(self, observations):
        return self.encoder(observations)

def make_env(rom_path, state_path, rank):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        env = MarioNdsEnv(rom_path=rom_path, state_path=state_path)
        env = Monitor(env)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train Mario RL Agent")
    parser.add_argument("--rom", type=str, default="../data/mario.nds", help="Path to NDS ROM")
    parser.add_argument("--state", type=str, default="../data/state.dst", help="Path to Savestate")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train")
    parser.add_argument("--test-run", action="store_true", help="Run a short test to verify environment")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing model to resume training (e.g. models/ppo_mario)")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments to run")
    parser.add_argument("--use-autoencoder", action="store_true", help="Use pre-trained Autoencoder for vision")
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.rom), exist_ok=True)
    if not os.path.exists(args.rom):
        print(f"Warning: ROM not found at {args.rom}. Please place it there.")
    if not os.path.exists(args.state):
        print(f"Warning: Savestate not found at {args.state}. Please place it there.")

    # Initialize MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_NDS_RL")

    # MLOps context
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "PPO")
        mlflow.log_param("total_timesteps", args.timesteps)
        mlflow.log_param("frameskip", 6)
        
        # Create parallel environments
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
        
        env_fns = [make_env(args.rom, args.state, i) for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns) # Launch multiple emulators in parallel!
        env = VecFrameStack(env, n_stack=4) # Stack 4 frames so CNN can perceive motion

        # Initialize or Load Model
        ent_coef_val = 0.05  # Increased entropy to force more exploration
        
        policy_kwargs = dict()
        if args.use_autoencoder:
            policy_kwargs = dict(
                features_extractor_class=CustomAutoencoderFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=512)
            )
            
        if args.resume and os.path.exists(f"{args.resume}.zip"):
            print(f"Resuming training from {args.resume}.zip...")
            model = PPO.load(args.resume, env=env, ent_coef=ent_coef_val)
        else:
            model = PPO("CnnPolicy", env, verbose=1, ent_coef=ent_coef_val, tensorboard_log="./tensorboard_logs/", policy_kwargs=policy_kwargs)
        
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("ent_coef", ent_coef_val)
        mlflow.log_param("n_steps", model.n_steps)
        mlflow.log_param("batch_size", model.batch_size)

        print("Starting training...")
        
        timesteps = 1000 if args.test_run else args.timesteps
        
        # Train Model with graceful interruption
        try:
            # If resuming, we tell SB3 NOT to reset the global step counter and learning rate schedule
            reset_ts = False if args.resume else True
            model.learn(total_timesteps=timesteps, callback=MLflowCallback(), reset_num_timesteps=reset_ts)
        except KeyboardInterrupt:
            print("\nTreinamento interrompido pelo usuário! Salvando o progresso atual...")

        # Save Model locally (this runs whether it finishes naturally or is interrupted)
        model_path = "models/ppo_mario"
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        
        # Register Model in MLflow
        mlflow.log_artifact(f"{model_path}.zip", artifact_path="models")
        
        print("Training complete and model saved.")

if __name__ == "__main__":
    main()
