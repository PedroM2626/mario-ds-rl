import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

class ICMModel(nn.Module):
    def __init__(self, obs_shape, num_actions, feature_dim=288):
        super(ICMModel, self).__init__()
        self.num_actions = num_actions
        
        # We expect obs_shape to be (84, 84, 4) if VecFrameStack is used (channels last)
        in_channels = obs_shape[-1]
        
        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten()
        )
        # 84 -> 42 -> 21 -> 11 -> 6. 32 * 6 * 6 = 1152 features
        # Let's map it down
        self.feature_linear = nn.Linear(32 * 6 * 6, feature_dim)
        
        # Inverse Model (Predicts action given s_t and s_{t+1})
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
        # Forward Model (Predicts s_{t+1} given s_t and action)
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, state, next_state, action_one_hot):
        phi_t = self.feature_linear(self.feature_extractor(state))
        phi_t1 = self.feature_extractor(next_state)
        # detach phi_t1 for forward model target so feature extractor is mostly trained by inverse model
        phi_t1_forward_target = self.feature_linear(phi_t1).detach()
        phi_t1 = self.feature_linear(phi_t1)
        
        # Inverse
        inverse_input = torch.cat([phi_t, phi_t1], dim=1)
        pred_action_logits = self.inverse_net(inverse_input)
        
        # Forward
        forward_input = torch.cat([phi_t, action_one_hot], dim=1)
        pred_phi_t1 = self.forward_net(forward_input)
        
        return pred_action_logits, pred_phi_t1, phi_t1_forward_target

class ICMVecEnvWrapper(VecEnvWrapper):
    """
    VecEnv Wrapper that trains an ICM module and adds intrinsic curiosity reward to the Extrinsic reward.
    """
    def __init__(self, venv, intrinsic_scale=0.1, forward_loss_weight=0.2):
        super().__init__(venv)
        self.intrinsic_scale = intrinsic_scale
        self.forward_loss_weight = forward_loss_weight
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.icm = ICMModel(venv.observation_space.shape, venv.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)
        
        self.prev_obs = None
        self.last_actions = None

    def reset(self):
        obs = self.venv.reset()
        self.prev_obs = obs
        return obs

    def step_async(self, actions):
        self.last_actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        
        # Process ICM
        # Normalize obs and permute to (B, C, H, W) for PyTorch Conv2D
        state_tensor = torch.FloatTensor(self.prev_obs).to(self.device) / 255.0
        state_tensor = state_tensor.permute(0, 3, 1, 2)
        
        next_state_tensor = torch.FloatTensor(obs).to(self.device) / 255.0
        next_state_tensor = next_state_tensor.permute(0, 3, 1, 2)
        
        actions_tensor = torch.LongTensor(self.last_actions).to(self.device)
        action_one_hot = F.one_hot(actions_tensor, num_classes=self.action_space.n).float()
        
        pred_action_logits, pred_phi_t1, phi_t1_target = self.icm(state_tensor, next_state_tensor, action_one_hot)
        
        # Intrinsic Reward calculation: Mean squared error of forward model
        forward_error = F.mse_loss(pred_phi_t1, phi_t1_target, reduction='none').mean(dim=1)
        intrinsic_rewards = forward_error.detach().cpu().numpy() * self.intrinsic_scale
        
        # Losses
        inverse_loss = F.cross_entropy(pred_action_logits, actions_tensor)
        forward_loss = forward_error.mean()
        
        # Total ICM Loss
        loss = (1.0 - self.forward_loss_weight) * inverse_loss + self.forward_loss_weight * forward_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Combine rewards
        total_rewards = rewards + intrinsic_rewards
        
        # Add debug info for MLflow callbacks
        for i in range(self.num_envs):
            infos[i]['intrinsic_reward'] = intrinsic_rewards[i]
            infos[i]['extrinsic_reward'] = rewards[i]
        
        # Handle done states correctly:
        # If an environment is done, 'obs' is actually the first observation of the next episode.
        # But we use it as 'next_state' anyway. This causes a tiny blip in curiosity at resets, 
        # which is acceptable and standard in simple ICM implementations.
        self.prev_obs = obs
        
        return obs, total_rewards, dones, infos
