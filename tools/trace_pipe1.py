import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))
from desmume.emulator import DeSmuME
from desmume.controls import Keys, keymask
from src.ram_env import MarioRamEnv
from src.eval_pinn_realtime import ReactiveAgent, SPAWN_ABS_PX, TilemapAStar

planner = TilemapAStar()
agent = ReactiveAgent(planner=planner)
env = MarioRamEnv(geo=True, max_steps=100)
obs, _ = env.reset()

print("Step-by-step trace near Pipe 1 (384px - 460px):")
for step in range(1, 35):
    act = agent.act(obs)
    obs, r, term, trunc, info = env.step(act)
    abs_px = SPAWN_ABS_PX + obs[0] * 512.0
    y_rel = obs[1] * 512.0
    vy = obs[3]
    on_ground = obs[4]
    ens = info.get("enemies", [])
    en_desc = [(e.get("name"), round(e.get("dx", 0)), round(e.get("dy", 0))) for e in ens[:2]]
    if abs_px > 300:
        print(f"step {step:2d} | act {act} | x={abs_px:.1f} y={y_rel:.1f} vy={vy:.2f} g={on_ground:.0f} | clearing={agent.clearing} | enemies: {en_desc}")
    if term:
        print(f"DIED at step {step}: abs_x={abs_px:.1f} died={info}")
        break

env.close()
