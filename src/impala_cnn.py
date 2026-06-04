import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaCNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        in_channels = observation_space.shape[0]
        # IMPALA CNN Architecture (OpenAI Procgen / IMPALA paper standard)
        self.block1 = ImpalaBlock(in_channels, 16)
        self.block2 = ImpalaBlock(16, 32)
        self.block3 = ImpalaBlock(32, 32)
        
        # Calculate feature map size after three maxpools (kernel 3, stride 2, padding 1):
        # 84 -> 42 -> 21 -> 11
        # Output spatial dimensions: 11x11
        # Channels: 32
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Standard SB3 observations are shape [batch_size, channels, height, width]
        x = self.block1(observations)
        x = self.block2(x)
        x = self.block3(x)
        return self.fc(x)
