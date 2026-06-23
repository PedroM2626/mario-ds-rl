import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import mlflow

def random_crop(x, out_size=84, padding=4):
    """
    Random crop data augmentation for PyTorch tensors.
    x has shape (B, C, H, W)
    """
    # Pad images on all sides
    x_padded = F.pad(x, [padding, padding, padding, padding], mode='replicate')
    B, C, H, W = x_padded.shape
    
    # Randomly select crop positions
    h_offsets = torch.randint(0, H - out_size + 1, (B,), device=x.device)
    w_offsets = torch.randint(0, W - out_size + 1, (B,), device=x.device)
    
    crops = torch.empty((B, C, out_size, out_size), device=x.device, dtype=x.dtype)
    for i in range(B):
        crops[i] = x_padded[i, :, h_offsets[i]:h_offsets[i]+out_size, w_offsets[i]:w_offsets[i]+out_size]
    return crops

class CURL(nn.Module):
    """
    CURL Contrastive Loss Module.
    Computes InfoNCE loss using a bilinear similarity matrix W.
    """
    def __init__(self, z_dim, encoder, encoder_target):
        super(CURL, self).__init__()
        self.encoder = encoder
        self.encoder_target = encoder_target
        # Learnable bilinear parameter W
        self.W = nn.Parameter(torch.randn(z_dim, z_dim))
        
    def compute_logits(self, z_a, z_k):
        # Bilinear product similarity: z_a W z_k^T
        Wz = torch.matmul(z_a, self.W) # (B, z_dim)
        logits = torch.matmul(Wz, z_k.T) # (B, B)
        # Subtract max for numerical stability
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        return logits

class CURLCallback(BaseCallback):
    """
    Stable Baselines 3 Callback to train the features extractor using CURL (contrastive loss).
    """
    def __init__(
        self,
        curl_lr=1e-4,
        batch_size=64,
        epochs=5,
        tau=0.05,
        verbose=0
    ):
        super(CURLCallback, self).__init__(verbose)
        self.curl_lr = curl_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.tau = tau
        
        self.curl = None
        self.optimizer = None
        
    def _on_training_start(self) -> None:
        # Get the query encoder from the policy
        query_encoder = self.model.policy.features_extractor
        
        # Verify that encoder is trainable
        trainable_params = sum(p.numel() for p in query_encoder.parameters() if p.requires_grad)
        if trainable_params == 0:
            print("[CURL] Warning: The features extractor (encoder) parameters are frozen. CURL will only train W.")
            
        # Create key (target) encoder as a deepcopy of the query encoder
        import copy
        self.key_encoder = copy.deepcopy(query_encoder).to(self.model.device)
        
        # Freeze key encoder parameters
        for p in self.key_encoder.parameters():
            p.requires_grad = False
            
        # Initialize CURL module
        # The latent dimension size matches the output features_dim
        z_dim = query_encoder.features_dim
        self.curl = CURL(z_dim, query_encoder, self.key_encoder).to(self.model.device)
        
        # Optimizer for CURL parameters (bilinear matrix W and query encoder parameters)
        params = list(self.curl.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.curl_lr)
        
        print(f"[CURL] Initialized on {self.model.device}. Learnable parameters count: {sum(p.numel() for p in params if p.requires_grad)}")

    def _on_step(self) -> bool:
        # CURL updates run at the end of each rollout
        return True

    def _on_rollout_end(self) -> None:
        # Extract observations from the rollout buffer
        obs = self.model.rollout_buffer.observations
        
        # Convert to float tensor and send to device
        obs_tensor = torch.as_tensor(obs, device=self.model.device).float()
        
        # Ensure channel-first format (B, C, H, W)
        if obs_tensor.shape[-1] == 1:
            obs_tensor = obs_tensor.permute(0, 1, 4, 2, 3)
            
        # Flatten sequence and environment dimensions
        # Shape: (n_steps * num_envs, channels, height, width)
        obs_flat = obs_tensor.reshape(-1, *obs_tensor.shape[2:])
        
        # Normalize to [0.0, 1.0] if range is [0, 255]
        if obs_flat.max() > 1.0:
            obs_flat = obs_flat / 255.0
            
        total_obs = obs_flat.shape[0]
        if total_obs < self.batch_size:
            return
            
        losses = []
        
        # Perform contrastive epochs
        for epoch in range(self.epochs):
            # Sample batch indices
            indices = torch.randperm(total_obs)[:self.batch_size]
            batch_obs = obs_flat[indices]
            
            # Generate two augmented views (anchors and keys) via random crop
            obs_anchor = random_crop(batch_obs, out_size=84, padding=4)
            obs_key = random_crop(batch_obs, out_size=84, padding=4)
            
            # Forward pass: encode views
            z_a = self.curl.encoder(obs_anchor)
            with torch.no_grad():
                z_k = self.curl.encoder_target(obs_key)
                
            # Compute logits and InfoNCE loss
            logits = self.curl.compute_logits(z_a, z_k)
            labels = torch.arange(self.batch_size, device=self.model.device)
            loss = F.cross_entropy(logits, labels)
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
            
            # Update key (target) encoder weights via EMA
            with torch.no_grad():
                for param, param_target in zip(self.curl.encoder.parameters(), self.curl.encoder_target.parameters()):
                    param_target.data.copy_(self.tau * param.data + (1.0 - self.tau) * param_target.data)
                    
        mean_loss = np.mean(losses)
        
        # Log contrastive loss to MLflow
        if mlflow.active_run():
            mlflow.log_metric("curl_loss", mean_loss, step=self.num_timesteps)
            
        if self.verbose > 0:
            print(f"[CURL] Step {self.num_timesteps} - mean loss: {mean_loss:.4f}")
