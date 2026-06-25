import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import mlflow
from curl import random_crop

def safe_cosine_similarity(x1, x2, dim=-1, eps=1e-6):
    # Add eps to norm to prevent division by zero (forward) and gradient explosion (backward)
    x1_norm = x1.norm(dim=dim, keepdim=True) + eps
    x2_norm = x2.norm(dim=dim, keepdim=True) + eps
    return (x1 / x1_norm * x2 / x2_norm).sum(dim=dim)


class TransitionModel(nn.Module):
    """
    Predicts the next representation z_{t+1} given the current representation z_t and action a_t.
    """
    def __init__(self, z_dim: int, action_dim: int = 64, num_actions: int = 6):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, action_dim)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + action_dim, z_dim),
            nn.LayerNorm(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim)
        )
        
    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim)
        # action: (B,)
        a_emb = self.action_embed(action) # (B, action_dim)
        x = torch.cat([z, a_emb], dim=-1) # (B, z_dim + action_dim)
        return self.mlp(x)

class ProjectionHead(nn.Module):
    """
    Projects representation features to a lower dimensional space.
    """
    def __init__(self, z_dim: int, proj_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class PredictionHead(nn.Module):
    """
    Predicts the projected target representation.
    """
    def __init__(self, proj_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class SPR(nn.Module):
    """
    Self-Supervised Policy Representations (SPR) container module.
    """
    def __init__(self, z_dim: int, encoder: nn.Module, target_encoder: nn.Module, proj_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.projector = ProjectionHead(z_dim, proj_dim)
        self.target_projector = ProjectionHead(z_dim, proj_dim)
        self.predictor = PredictionHead(proj_dim)
        self.transition_model = TransitionModel(z_dim)

class SPRCallback(BaseCallback):
    """
    Callback that executes multi-step future latent prediction (SPR) to optimize representation learning.
    """
    def __init__(
        self,
        spr_lr: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 5,
        k_steps: int = 3, # number of future prediction steps
        tau: float = 0.05,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.spr_lr = spr_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_steps = k_steps
        self.tau = tau
        
        self.spr = None
        self.optimizer = None
        
    def _on_training_start(self) -> None:
        # Get query encoder
        query_encoder = self.model.policy.features_extractor
        
        # Verify that encoder is trainable
        trainable_params = sum(p.numel() for p in query_encoder.parameters() if p.requires_grad)
        if trainable_params == 0:
            print("[SPR] Warning: Features extractor is frozen. SPR will only train projection/transition models.")
            
        # Create target encoder
        import copy
        self.target_encoder = copy.deepcopy(query_encoder).to(self.model.device)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        # Initialize SPR container
        z_dim = query_encoder.features_dim
        self.spr = SPR(z_dim, query_encoder, self.target_encoder).to(self.model.device)
        
        # Freeze target projector
        for p in self.spr.target_projector.parameters():
            p.requires_grad = False
        # Copy projector weights to target projector
        self.spr.target_projector.load_state_dict(self.spr.projector.state_dict())
        
        # Setup optimizer for all parameters (projector, predictor, transition_model, and encoder)
        params = list(self.spr.parameters())
        # Filter out target weights to keep it clean
        trainable_params = [p for p in params if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.spr_lr)
        
        print(f"[SPR] Initialized on {self.model.device}. Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Extract sequences from the rollout buffer
        obs = self.model.rollout_buffer.observations # (n_steps, num_envs, C, H, W)
        actions = self.model.rollout_buffer.actions # (n_steps, num_envs)
        
        # Convert to torch tensors
        obs_tensor = torch.as_tensor(obs, device=self.model.device).float()
        actions_tensor = torch.as_tensor(actions, device=self.model.device).long()
        if len(actions_tensor.shape) == 3:
            actions_tensor = actions_tensor.squeeze(-1)
        
        n_steps, num_envs = actions_tensor.shape
        K = self.k_steps
        
        if n_steps <= K:
            return
            
        # Ensure channel-first format (n_steps, num_envs, C, H, W)
        if obs_tensor.shape[-1] == 1:
            obs_tensor = obs_tensor.permute(0, 1, 4, 2, 3)
            
        losses = []
        
        # Otimização por épocas
        for epoch in range(self.epochs):
            # Sample random starts along the sequence and environment indices
            t_indices = torch.randint(0, n_steps - K, (self.batch_size,))
            env_indices = torch.randint(0, num_envs, (self.batch_size,))
            
            # Slice observations and actions
            batch_obs = [] # list of length K+1, each tensor shape (B, C, H, W)
            batch_actions = [] # list of length K, each tensor shape (B,)
            
            for k in range(K + 1):
                o = obs_tensor[t_indices + k, env_indices]
                if o.max() > 1.0:
                    o = o / 255.0
                batch_obs.append(o)
                
            for k in range(K):
                a = actions_tensor[t_indices + k, env_indices]
                batch_actions.append(a)
                
            # Apply independent random crop augmentations to online and target views
            batch_obs_online = [random_crop(o, padding=4) for o in batch_obs]
            batch_obs_target = [random_crop(o, padding=4) for o in batch_obs]
            
            # 1. Step 0: Encode first observation
            z = self.spr.encoder(batch_obs_online[0])
            
            # Compute projection & prediction
            p0 = self.spr.predictor(self.spr.projector(z))
            
            # Target projection
            with torch.no_grad():
                z_target0 = self.spr.target_encoder(batch_obs_target[0])
                p_target0 = self.spr.target_projector(z_target0)
                
            # Cosine similarity loss at step 0
            loss = - safe_cosine_similarity(p0, p_target0, dim=-1).mean()
            
            # 2. Steps 1 to K: Rollout transition model
            for k in range(K):
                # Transition step: z_{t+k+1} = g(z_{t+k}, a_{t+k})
                z = self.spr.transition_model(z, batch_actions[k])
                
                # Predict future projection
                p = self.spr.predictor(self.spr.projector(z))
                
                # Target future projection
                with torch.no_grad():
                    z_target = self.spr.target_encoder(batch_obs_target[k+1])
                    p_target = self.spr.target_projector(z_target)
                    
                # Add to multi-step loss
                loss += - safe_cosine_similarity(p, p_target, dim=-1).mean()
                
            # Average loss over all prediction steps
            loss = loss / (K + 1)
            
            # Gradient update
            self.optimizer.zero_grad()
            loss.backward()
            
            # Check for NaNs or Infs in gradients to prevent model corruption
            trainable_params = [p for p in self.spr.parameters() if p.requires_grad]
            has_nan = False
            for p in trainable_params:
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan = True
                    break
            
            if has_nan:
                print(f"[SPR] Warning: NaN/Inf gradients detected at step {self.num_timesteps}. Skipping optimizer step.")
                self.optimizer.zero_grad()
            else:
                # Apply gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                self.optimizer.step()
            
            losses.append(loss.item())
            
            # Update targets via EMA
            with torch.no_grad():
                for param, param_target in zip(self.spr.encoder.parameters(), self.spr.target_encoder.parameters()):
                    param_target.data.copy_(self.tau * param.data + (1.0 - self.tau) * param_target.data)
                for param, param_target in zip(self.spr.projector.parameters(), self.spr.target_projector.parameters()):
                    param_target.data.copy_(self.tau * param.data + (1.0 - self.tau) * param_target.data)
                    
        mean_loss = np.mean(losses)
        
        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_metric("spr_loss", mean_loss, step=self.num_timesteps)
            
        if self.verbose > 0:
            print(f"[SPR] Step {self.num_timesteps} - mean loss: {mean_loss:.4f}")
