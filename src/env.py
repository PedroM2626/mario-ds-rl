import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os

try:
    from desmume.emulator import DeSmuME
    from desmume.controls import Keys
except ImportError:
    print("Warning: py-desmume is not installed or failed to load.")
    # Provide dummy classes for testing without emulator
    class Keys:
        A = 1; B = 2; X = 4; Y = 8; UP = 16; DOWN = 32; LEFT = 64; RIGHT = 128
        START = 256; SELECT = 512; L = 1024; R = 2048

class MarioNdsEnv(gym.Env):
    """
    Custom Environment that follows gym interface for playing New Super Mario Bros DS.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, rom_path, state_path, render_mode=None):
        super(MarioNdsEnv, self).__init__()
        
        self.rom_path = rom_path
        self.state_path = state_path
        self.render_mode = render_mode
        
        # Initialize Emulator
        try:
            self.emu = DeSmuME()
            self.emu.open(self.rom_path)
            self.emu.savestate.load_file(self.state_path)
            self.has_emulator = True
        except Exception as e:
            print(f"Failed to initialize emulator: {e}")
            self.has_emulator = False

        # Define action space
        # Actions: 0: Noop, 1: Right, 2: Right+Dash(B), 3: Right+Dash+Jump(A), 4: Left, 5: Jump
        self.action_space = spaces.Discrete(6)
        
        # Define observation space (RGB image of the top screen: 256x192)
        # We will downscale it to 84x84 and convert to grayscale for CNN efficiency
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(84, 84, 1), dtype=np.uint8)

        self.current_x = 0
        self.frameskip = 4

    def _get_obs(self):
        if not self.has_emulator:
            return np.zeros((84, 84, 1), dtype=np.uint8)
            
        # Get frame from top screen (or bottom, depending on where gameplay is)
        # In NSMB DS, gameplay is mostly on the top screen
        frame = self.emu.display_buffer_as_rgbx() # This might need adjustment based on py-desmume API
        # py-desmume display_buffer returns raw pixels. We assume a method or reshape.
        # For this skeleton, we handle it generally:
        try:
            # The DS screen is 256x384 total (two 256x192 screens)
            # frame is usually a flat array or 256x384x4
            frame = np.array(frame, dtype=np.uint8).reshape((384, 256, 4))
            top_screen = frame[:192, :, :3] # Take top half, RGB only
            
            # Convert to grayscale and resize
            gray = cv2.cvtColor(top_screen, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            return np.expand_dims(resized, axis=-1)
        except Exception:
            return np.zeros((84, 84, 1), dtype=np.uint8)

    def _get_info(self):
        return {"x_position": self.current_x}

    def _get_mario_x(self):
        # NOTE: This requires the exact RAM address for Mario's X position.
        # For NSMB DS, memory is dynamically allocated, so you might need to find a pointer.
        # This is a placeholder. If using pure pixels for reward, you'd calculate it differently.
        if not self.has_emulator:
            return self.current_x + 1 # Dummy increment
            
        try:
            # Example reading from RAM (address is fake)
            # val = self.emu.memory.read_unsigned_short(0x02000000)
            val = 0 # Placeholder
            return val
        except:
            return self.current_x

    def step(self, action):
        if not self.has_emulator:
            self.current_x += 1
            return self._get_obs(), 1.0, self.current_x > 100, False, self._get_info()

        # Map actions to emulator keys
        keys = []
        if action == 1:
            keys.append(Keys.RIGHT)
        elif action == 2:
            keys.append(Keys.RIGHT)
            keys.append(Keys.B)
        elif action == 3:
            keys.append(Keys.RIGHT)
            keys.append(Keys.B)
            keys.append(Keys.A)
        elif action == 4:
            keys.append(Keys.LEFT)
        elif action == 5:
            keys.append(Keys.A)

        # Apply inputs and run frameskip
        for key in keys:
            self.emu.input.keypad_add_key(key)
            
        for _ in range(self.frameskip):
            self.emu.cycle()
            
        for key in keys:
            self.emu.input.keypad_rm_key(key)

        # Get observation
        obs = self._get_obs()
        
        # Calculate Reward (e.g., based on moving right)
        new_x = self._get_mario_x()
        reward = new_x - self.current_x
        self.current_x = new_x
        
        # Check if done (e.g., Mario dies or wins)
        # You would read a RAM address to check for death (e.g., 0x0208B364 for lives)
        done = False 
        truncated = False
        info = self._get_info()

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.has_emulator:
            self.emu.savestate.load_file(self.state_path)
            
        self.current_x = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def render(self):
        if self.render_mode == 'human':
            pass # Implement rendering with OpenCV if needed
        elif self.render_mode == 'rgb_array':
            return self._get_obs()

    def close(self):
        if self.has_emulator:
            self.emu.destroy()
