import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os

try:
    from desmume.emulator import DeSmuME
    from desmume.controls import Keys, keymask
except ImportError:
    print("Warning: py-desmume is not installed or failed to load.")
    # Provide dummy classes for testing without emulator
    class Keys:
        KEY_A = 1; KEY_B = 2; KEY_X = 4; KEY_Y = 8; KEY_UP = 16; KEY_DOWN = 32; KEY_LEFT = 64; KEY_RIGHT = 128
        KEY_START = 256; KEY_SELECT = 512; KEY_L = 1024; KEY_R = 2048
        
    def keymask(k):
        return k

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
        
        # Load death template mask if available
        self.death_mask = None
        # Try to find images/death.png relative to the project root
        death_img_path = os.path.abspath(os.path.join(os.path.dirname(self.rom_path), '../images/death.png'))
        if os.path.exists(death_img_path):
            d_img = cv2.imread(death_img_path)
            if d_img is not None:
                d_gray = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
                d_resized = cv2.resize(d_gray, (84, 84), interpolation=cv2.INTER_AREA)
                self.death_mask = (d_resized < 10)
                print("Loaded death.png mask for Game Over detection.")
        
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
        # Ações restauradas para o controle total do Mario
        # Actions: 0: Noop, 1: Right, 2: Right+Dash(B), 3: Right+Dash+Jump(B+A), 4: Left, 5: Jump(A)
        self.action_space = spaces.Discrete(6)
        
        # Define observation space (RGB image of the top screen: 256x192)
        # We will downscale it to 84x84 and convert to grayscale for CNN efficiency
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(84, 84, 1), dtype=np.uint8)

        # Optical flow needs previous frame
        self.prev_gray = None
        self.current_x = 0
        self.frameskip = 6 # Aumentado de 4 para 6

    def _get_obs(self):
        if not self.has_emulator:
            return np.zeros((84, 84, 1), dtype=np.uint8)
            
        try:
            # We assume display_buffer_as_rgbx returns a flat array of 256x384x4 (RGBA)
            frame = self.emu.display_buffer_as_rgbx()
            frame = np.array(frame, dtype=np.uint8).reshape((384, 256, 4))
            top_screen = frame[:192, :, :3] # Take top half, RGB only
            
            # Convert to grayscale and resize
            gray = cv2.cvtColor(top_screen, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            
            # Optical Flow reward calculation: track the background scrolling
            # If background moves left, Mario is moving right
            reward_displacement = 0.0
            if self.prev_gray is not None:
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(self.prev_gray, resized, None, 
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                # Average horizontal flow (flow[..., 0])
                avg_flow_x = np.mean(flow[..., 0])
                
                # Convert flow to displacement (negative flow means Mario moved right)
                if avg_flow_x < -0.2:  
                    flow_disp = -avg_flow_x
                elif avg_flow_x > 0.2:
                    flow_disp = -avg_flow_x # This will be negative since avg_flow_x is positive
                else:
                    flow_disp = 0.0
                    
                self.accumulated_x += flow_disp
                
                # ONLY reward Mario if he reaches a new record distance in this episode
                if self.accumulated_x > self.max_x:
                    reward_displacement = (self.accumulated_x - self.max_x) * 5.0
                    self.max_x = self.accumulated_x
                else:
                    reward_displacement = 0.0
                
            self.prev_gray = resized.copy()
            self.last_reward_displacement = reward_displacement
            
            return np.expand_dims(resized, axis=-1)
        except Exception:
            self.last_reward_displacement = 0.0
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
            keys.append(Keys.KEY_RIGHT)
        elif action == 2:
            keys.append(Keys.KEY_RIGHT)
            keys.append(Keys.KEY_B)
        elif action == 3:
            keys.append(Keys.KEY_RIGHT)
            keys.append(Keys.KEY_B)
            keys.append(Keys.KEY_A)
        elif action == 4:
            keys.append(Keys.KEY_LEFT)
        elif action == 5:
            keys.append(Keys.KEY_A)

        # Apply inputs and run frameskip
        for key in keys:
            self.emu.input.keypad_add_key(keymask(key))
            
        for _ in range(self.frameskip):
            self.emu.cycle()
            
        for key in keys:
            self.emu.input.keypad_rm_key(keymask(key))

        # Get observation (this also updates self.last_reward_displacement)
        obs = self._get_obs()
        
        # Calculate Reward based on computer vision optical flow
        if not self.has_emulator:
            reward = 1.0
        else:
            # We use the displacement calculated in _get_obs
            # Reduzimos a penalidade de tempo para evitar o suicídio intencional, 
            # mas ainda forçamos ele a não ficar parado para sempre.
            time_penalty = -0.05
            reward = self.last_reward_displacement + time_penalty
            self.current_x = self.accumulated_x # Track approximate distance for info
        
        # Check if done (e.g., Mario dies or wins)
        done = False 
        truncated = False
        
        # Add timeout to prevent infinite standing still episodes
        # 1 step = 6 frames. 60 frames = 1 sec. 1 step = 1/10 sec.
        # 2m50s = 170 segundos. 170 * 10 = 1700 steps.
        self.episode_steps += 1
        if self.episode_steps >= 1700:
            truncated = True
        
        if self.death_mask is not None:
            # Check if the current observation matches the black Bowser silhouette
            obs_2d = np.squeeze(obs)
            # Calculate what percentage of the expected black pixels are actually black
            black_match_ratio = np.mean(obs_2d[self.death_mask] < 10)
            
            if black_match_ratio > 0.90:  # 90% of the mask matches
                done = True
                reward -= 30.0  
                print("Death detected!")
                
        info = self._get_info()

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.has_emulator:
            self.emu.savestate.load_file(self.state_path)
            
        self.accumulated_x = 0.0
        self.max_x = 0.0
        self.episode_steps = 0
        self.prev_gray = None
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def render(self):
        if self.render_mode == 'human':
            obs = self._get_obs()
            if obs is not None:
                # Rescale 84x84 up for better visibility
                display_img = cv2.resize(obs, (336, 336), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Mario DS RL", display_img)
                cv2.waitKey(1)
        elif self.render_mode == 'rgb_array':
            return self._get_obs()

    def close(self):
        if self.has_emulator:
            self.emu.destroy()
