import gymnasium as gym
import numpy as np
import os

class MarioNDSNEDreamerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
        img_shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), bool),
            "is_last": gym.spaces.Box(0, 1, (), bool),
            "is_terminal": gym.spaces.Box(0, 1, (), bool),
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_dict = {
            "image": obs,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs_dict

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        obs_dict = {
            "image": obs,
            "is_first": False,
            "is_last": done,
            "is_terminal": terminated,
        }
        return obs_dict, reward, done, info

def make_nedreamer_env(**kwargs):
    from env import MarioNdsEnv
    
    # Resolve relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rom_path = os.path.join(base_dir, "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
    state_path = os.path.join(base_dir, "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")
    
    render_mode = os.getenv("MARIO_RENDER_MODE", None)
    raw_env = MarioNdsEnv(rom_path=rom_path, state_path=state_path, render_mode=render_mode)
    
    return MarioNDSNEDreamerWrapper(raw_env)
