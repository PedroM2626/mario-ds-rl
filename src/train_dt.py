import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
from env import MarioNdsEnv
from decision_transformer import DecisionTransformer

class DecisionTransformerDataset(Dataset):
    """
    Custom PyTorch Dataset for loading trajectories and preparing sequences
    for training the Decision Transformer.
    """
    def __init__(self, trajectories_path, context_len=30, scale=1.0):
        with open(trajectories_path, "rb") as f:
            self.trajectories = pickle.load(f)
            
        self.context_len = context_len
        self.scale = scale # Scale factor for RTG to aid optimization stability
        
        # Calculate dataset statistics
        self.num_episodes = len(self.trajectories)
        self.total_steps = sum(len(traj["rewards"]) for traj in self.trajectories)
        
        # Calculate Returns-To-Go (RTG) for each trajectory
        for traj in self.trajectories:
            rewards = traj["rewards"]
            # Retroactively calculate returns-to-go
            rtg = np.zeros_like(rewards)
            running_return = 0
            for t in reversed(range(len(rewards))):
                running_return += rewards[t]
                rtg[t] = running_return
            traj["rtgs"] = rtg
            
        print(f"Loaded {self.num_episodes} trajectories containing {self.total_steps} total transition steps.")

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = len(traj["rewards"])
        
        # Randomly sample a starting point in the episode trajectory
        if traj_len > self.context_len:
            start_idx = np.random.randint(0, traj_len - self.context_len + 1)
            end_idx = start_idx + self.context_len
            
            # Slice sequence
            states = traj["observations"][start_idx:end_idx] # (context_len, 84, 84, 1)
            actions = traj["actions"][start_idx:end_idx]     # (context_len,)
            rtgs = traj["rtgs"][start_idx:end_idx]           # (context_len,)
            timesteps = np.arange(start_idx, end_idx)         # (context_len,)
            
            # Mask is fully valid (1s)
            attention_mask = np.ones(self.context_len, dtype=np.float32)
        else:
            # Trajectory is shorter than context length; padding required
            states = traj["observations"]
            actions = traj["actions"]
            rtgs = traj["rtgs"]
            timesteps = np.arange(0, traj_len)
            
            pad_len = self.context_len - traj_len
            
            # Pad sequences (pre-padding or post-padding; post-padding is common in DT)
            # states padding with zeros
            padding_states = np.zeros((pad_len, 84, 84, 1), dtype=np.uint8)
            states = np.concatenate([states, padding_states], axis=0)
            
            # actions padding with 0 (Noop)
            padding_actions = np.zeros(pad_len, dtype=np.int64)
            actions = np.concatenate([actions, padding_actions], axis=0)
            
            # rtgs padding with 0
            padding_rtgs = np.zeros(pad_len, dtype=np.float32)
            rtgs = np.concatenate([rtgs, padding_rtgs], axis=0)
            
            # timesteps padding with the last timestep index
            padding_timesteps = np.full(pad_len, traj_len - 1, dtype=np.int64)
            timesteps = np.concatenate([timesteps, padding_timesteps], axis=0)
            
            # Mask: 1 for real data, 0 for padded data
            attention_mask = np.concatenate([np.ones(traj_len, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)], axis=0)
            
        # Reformat states to channel-first format (context_len, 1, 84, 84)
        states = states.transpose(0, 3, 1, 2) # (context_len, 1, 84, 84)
        
        # Apply scaling to RTG
        rtgs = rtgs * self.scale
        
        # Convert to PyTorch tensors
        return (
            torch.tensor(states, dtype=torch.uint8),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rtgs, dtype=torch.float32).unsqueeze(-1), # (context_len, 1)
            torch.tensor(timesteps, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float32)
        )

def evaluate_online(model, rom_path, state_path, context_len, target_return=300.0, scale=1.0, device="cpu"):
    """
    Evaluates the trained Decision Transformer online in the actual game environment.
    """
    print(f"\n[Eval] Starting online evaluation with target return: {target_return}...")
    
    import shutil
    temp_rom = "data/mario_eval_temp.nds"
    temp_state = "data/state_eval_temp.ds1"
    
    print("Creating temporary copies of ROM and Savestate for evaluation to avoid file locking...")
    shutil.copyfile(rom_path, temp_rom)
    shutil.copyfile(state_path, temp_state)
    
    try:
        env = MarioNdsEnv(rom_path=temp_rom, state_path=temp_state)
        
        obs, _ = env.reset()
        rtg = target_return
        
        # Format initial observation: (1, 84, 84)
        obs_formatted = obs.transpose(2, 0, 1) # (1, 84, 84)
        
        # Store history for autoregressive conditioning
        states_hist = [obs_formatted]
        actions_hist = []
        rtgs_hist = [rtg * scale]
        timesteps_hist = [0]
        
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0
        
        model.eval()
        
        with torch.no_grad():
            while not (done or truncated):
                # Form sequences of max length context_len
                t_states = torch.tensor(states_hist[-context_len:], dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                
                # For actions, we need padding at the beginning of the episode if history is empty
                if len(actions_hist) == 0:
                    t_actions = torch.zeros((1, 1), dtype=torch.long, device=device)
                    # Expand states to match sequence length of 1
                    t_rtgs = torch.tensor(rtgs_hist[-context_len:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
                    t_timesteps = torch.tensor(timesteps_hist[-context_len:], dtype=torch.long, device=device).unsqueeze(0)
                else:
                    # We need states and actions to align: if we have T states, we have T-1 actions
                    # So we pad actions with a dummy action at the end to make shapes match
                    actions_slice = actions_hist[-context_len:]
                    # Pad actions to match the length of states
                    if len(actions_slice) < len(states_hist[-context_len:]):
                        actions_slice = actions_slice + [0] # append dummy action
                    t_actions = torch.tensor(actions_slice, dtype=torch.long, device=device).unsqueeze(0)
                    t_rtgs = torch.tensor(rtgs_hist[-context_len:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
                    t_timesteps = torch.tensor(timesteps_hist[-context_len:], dtype=torch.long, device=device).unsqueeze(0)
                    
                # Predict action logits
                logits = model(t_states, t_actions, t_rtgs, t_timesteps)
                
                # Predict action for the last timestep (greedy selection)
                action_logits = logits[0, -1, :]
                action = torch.argmax(action_logits).item()
                
                # Step the environment
                next_obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                # Update history
                next_obs_formatted = next_obs.transpose(2, 0, 1)
                states_hist.append(next_obs_formatted)
                actions_hist.append(action)
                
                rtg = rtg - reward
                rtgs_hist.append(rtg * scale)
                timesteps_hist.append(steps)
                
                if steps % 100 == 0:
                    print(f"  [Eval] Step {steps} - Current Reward: {total_reward:.1f}, Current RTG: {rtg:.1f}")
                    
                # Max steps safeguard for evaluation
                if steps >= 1000:
                    break
    finally:
        print("Closing evaluation environment...")
        if 'env' in locals():
            env.close()
            
        print("Cleaning up evaluation temporary ROM and state copies...")
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
                
    print(f"[Eval] Completed! Total Steps: {steps}, Total Reward: {total_reward:.1f}")
    return total_reward, steps

def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer offline on collected trajectories")
    parser.add_argument("--rom", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds", help="Path to NDS ROM")
    parser.add_argument("--state", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1", help="Path to Savestate")
    parser.add_argument("--dataset-path", type=str, default="data/mario_trajectories.pkl", help="Path to collected trajectories")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--context-len", type=int, default=30, help="Context sequence length (T)")
    parser.add_argument("--rtg-scale", type=float, default=0.01, help="Scaling factor for returns-to-go")
    parser.add_argument("--target-eval-return", type=float, default=300.0, help="Target return for online evaluation")
    parser.add_argument("--hidden-size", type=int, default=256, help="Transformer hidden layer size")
    parser.add_argument("--n-layer", type=int, default=3, help="Number of Transformer layers")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--test-run", action="store_true", help="Perform a short test run to verify the pipeline")
    parser.add_argument("--run-id", type=str, default="decision_transformer_mario", help="Name of this run")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use for training")
    args = parser.parse_args()

    # Verify if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Trajectory dataset not found at {args.dataset_path}.")
        print("Please run collect_trajectories.py first to collect some gameplay data.")
        return

    # Check device
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_NDS_DecisionTransformer")

    # Load dataset & dataloader
    dataset = DecisionTransformerDataset(args.dataset_path, context_len=args.context_len, scale=args.rtg_scale)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Initialize Decision Transformer model
    try:
        model = DecisionTransformer(
            state_dim=512,
            act_dim=6,
            hidden_size=args.hidden_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            dropout=0.1
        ).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print("Warning: GPU CUDA Out Of Memory. Falling back to CPU for execution.")
            device = torch.device("cpu")
            model = DecisionTransformer(
                state_dim=512,
                act_dim=6,
                hidden_size=args.hidden_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                dropout=0.1
            ).to(device)
        else:
            raise e
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction="none") # Compute loss per token then mask

    # Train model
    with mlflow.start_run(run_name=args.run_id):
        # Log hyperparameters to MLflow
        mlflow.log_param("model_type", "DecisionTransformer")
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("context_len", args.context_len)
        mlflow.log_param("rtg_scale", args.rtg_scale)
        mlflow.log_param("hidden_size", args.hidden_size)
        mlflow.log_param("n_layers", args.n_layer)
        mlflow.log_param("n_heads", args.n_head)
        
        print("\nStarting training loop...")
        
        global_step = 0
        for epoch in range(args.epochs):
            model.train()
            epoch_losses = []
            
            for step, (states, actions, rtgs, timesteps, masks) in enumerate(dataloader):
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                timesteps = timesteps.to(device)
                masks = masks.to(device)
                
                # Forward pass
                # logits shape: (batch_size, context_len, act_dim)
                logits = model(states, actions, rtgs, timesteps, attention_mask=masks)
                
                # Compute masked CrossEntropyLoss
                # Flatten outputs: (B * T, act_dim)
                flat_logits = logits.reshape(-1, logits.shape[-1])
                flat_actions = actions.reshape(-1)
                
                raw_loss = loss_fn(flat_logits, flat_actions)
                
                # Mask out padding elements
                flat_masks = masks.reshape(-1)
                masked_loss = raw_loss * flat_masks
                
                # Average loss over only non-padded tokens
                loss = masked_loss.sum() / torch.clamp(flat_masks.sum(), min=1.0)
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                global_step += 1
                
                if step % 10 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
                    mlflow.log_metric("train_loss_step", loss.item(), step=global_step)
                    
                if args.test_run and step >= 2:
                    print("Test run completed early for verification!")
                    break
                    
            mean_loss = np.mean(epoch_losses)
            mlflow.log_metric("train_loss_epoch", mean_loss, step=epoch)
            print(f"--> Epoch {epoch+1} Complete | Mean Loss: {mean_loss:.4f}")
            
            # Periodically evaluate online
            if (epoch + 1) % max(1, args.epochs // 3) == 0 or args.test_run:
                eval_reward, eval_steps = evaluate_online(
                    model,
                    args.rom,
                    args.state,
                    args.context_len,
                    target_return=args.target_eval_return,
                    scale=args.rtg_scale,
                    device=device
                )
                mlflow.log_metric("eval_reward", eval_reward, step=epoch)
                mlflow.log_metric("eval_steps", eval_steps, step=epoch)
                
            if args.test_run:
                break
                
        # Save model
        os.makedirs("models", exist_ok=True)
        model_save_path = f"models/{args.run_id}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}!")
        mlflow.log_artifact(model_save_path, artifact_path="models")

if __name__ == "__main__":
    main()
