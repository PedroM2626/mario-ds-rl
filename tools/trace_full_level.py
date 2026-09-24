import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))
import time
from src.ram_env import MarioRamEnv
from src.eval_pinn_realtime import ReactiveAgent, SPAWN_ABS_PX, TilemapAStar

planner = TilemapAStar()
agent = ReactiveAgent(planner=planner)
env = MarioRamEnv(geo=True, max_steps=1200)
obs, _ = env.reset()

print("Tracing full level playthrough...")
last_x = 0
for step in range(1, 1201):
    act = agent.act(obs)
    obs, r, term, trunc, info = env.step(act)
    abs_px = SPAWN_ABS_PX + obs[0] * 512.0
    mario_ram_px = info.get("mario_px", 0.0)
    cleared = info.get("level_cleared", False)
    
    if step % 20 == 0 or step >= 100 or cleared or term or trunc:
        ex = [round(float(obs[8 + 3 * k]) * 256) for k in range(3)]
        et = [round(float(obs[10 + 3 * k]) * 256) for k in range(3)]
        ma_y = (env.ram._mario_pos()[0] / 4096.0) if env.ram.mario_base else 0.0
        print(f"step {step:4d} | x={abs_px:6.1f}px (ram_x={mario_ram_px:6.1f}px, y={ma_y:5.1f}) | act {act} | g={obs[4]:.0f} vy={obs[3]:5.2f} | edx={ex}")

    if abs_px >= 4016.0 or mario_ram_px >= 4016.0 or cleared:
        print(f"\n[CLEAR] LEVEL CLEARED AT STEP {step}! Final X: {abs_px:.1f}px (RAM X: {mario_ram_px:.1f}px)")
        break

    if term:
        cause = "death" if obs[6] > 0.02 else "timeout"
        ex = [round(float(obs[8 + 3 * k]) * 256) for k in range(3)]
        et = [round(float(obs[10 + 3 * k]) * 256) for k in range(3)]
        print(f"\n[TERM] TERMINATED ({cause}) at step {step}: abs_x={abs_px:.1f}px (RAM: {mario_ram_px:.1f}px)")
        print(f"   death diag: vy={obs[3]:.2f} ground={obs[4]:.0f} pit={[int(x) for x in obs[17:23]]} enemy_dx={ex} enemy_type={et}")
        break

    if trunc:
        print(f"\n[TRUNC] TRUNCATED at step {step}: max steps reached. Final X: {abs_px:.1f}px")
        break

env.close()
