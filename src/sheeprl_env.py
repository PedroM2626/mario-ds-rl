import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env import MarioNdsEnv

class SheepRLMarioWrapper(gym.Wrapper):
    """
    Wrapper to convert MarioNdsEnv observation space into a Gymnasium Dict space,
    transposing channels-last (84, 84, 1) to channels-first (1, 84, 84) for SheepRL.
    """
    def __init__(self, env, key="rgb"):
        super().__init__(env)
        self.key = key
        
        orig_space = env.observation_space
        # Transpose shape: (H, W, C) -> (C, H, W)
        new_shape = (orig_space.shape[2], orig_space.shape[0], orig_space.shape[1])
        
        self.observation_space = spaces.Dict({
            self.key: spaces.Box(
                low=0,
                high=255,
                shape=new_shape,
                dtype=np.uint8
            )
        })

    def _convert_obs(self, obs):
        # Transpose from (H, W, C) to (C, H, W)
        obs_transposed = obs.transpose(2, 0, 1)
        return {self.key: obs_transposed}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._convert_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_obs(obs), reward, terminated, truncated, info

def create_mario_dreamer_env(**kwargs):
    # Resolve relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_path = os.path.join(base_dir, "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
    state_path = os.path.join(base_dir, "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")
    
    render_mode = kwargs.get("render_mode", None)
    raw_env = MarioNdsEnv(rom_path=rom_path, state_path=state_path, render_mode=render_mode)
    return SheepRLMarioWrapper(raw_env)

# Register the environment with Gymnasium
gym.register(
    id="MarioNDS-Dreamer-v0",
    entry_point="sheeprl_env:create_mario_dreamer_env",
)
