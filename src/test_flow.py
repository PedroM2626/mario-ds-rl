import numpy as np
from env import MarioNdsEnv

def main():
    env = MarioNdsEnv(rom_path="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds", 
                      state_path="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1", 
                      render_mode=None)
    
    import cv2
    
    # Test Right (Action 1)
    obs, info = env.reset()
    print("\n--- Testing RIGHT (Action 1) ---")
    for i in range(5):
        obs, reward, done, truncated, info = env.step(1)
        print(f"Action: RIGHT, Reward: {reward:.4f}")
        cv2.imwrite(f"right_{i}.png", obs)
        
    # Test Jump (Action 5)
    obs, info = env.reset()
    print("\n--- Testing JUMP (Action 5) ---")
    for i in range(5):
        obs, reward, done, truncated, info = env.step(5)
        print(f"Action: JUMP, Reward: {reward:.4f}")
        cv2.imwrite(f"jump_{i}.png", obs)

    # Test Noop (Action 0)
    obs, info = env.reset()
    print("\n--- Testing NOOP (Action 0) ---")
    for _ in range(5):
        obs, reward, done, truncated, info = env.step(0)
        print(f"Action: NOOP, Reward: {reward:.4f}")

    env.close()

if __name__ == "__main__":
    main()
