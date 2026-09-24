import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))
from src.ram_env import MarioRamEnv
from src.eval_pinn_realtime import ReactiveAgent, SPAWN_ABS_PX, TilemapAStar

planner = TilemapAStar()
agent = ReactiveAgent(planner=planner)
env = MarioRamEnv(geo=True, max_steps=130)
obs, _ = env.reset()

print("Tracing steps 115-128 (2200px - 2350px):")
for step in range(1, 130):
    act = agent.act(obs)
    obs, r, term, trunc, info = env.step(act)
    abs_px = SPAWN_ABS_PX + obs[0] * 512.0
    if abs_px > 2200:
        ens = info.get("enemies", [])
        en_desc = [(e.get("name"), round(e.get("dx", 0)), round(e.get("dy", 0)), e.get("type")) for e in ens[:3]]
        pit = [int(x) for x in obs[17:23]]
        print(f"step {step:3d} | act {act} | x={abs_px:6.1f} y={obs[1]*512:5.1f} vy={obs[3]:5.2f} g={obs[4]:.0f} | clearing={agent.clearing} | pit={pit} | enemies: {en_desc}")
    if term:
        print(f"Died at step {step}: abs_x={abs_px:.1f}")
        break

env.close()
