import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))
from src.ram_env import MarioRamEnv
from src.eval_pinn_realtime import ReactiveAgent, SPAWN_ABS_PX, TilemapAStar

planner = TilemapAStar()
agent = ReactiveAgent(planner=planner)
env = MarioRamEnv(geo=True, max_steps=145)
obs, _ = env.reset()

print("Tracing steps 135-144 around Pit 6:")
for step in range(1, 145):
    act = agent.act(obs)
    # Check what agent's internal triggers saw
    surf = agent._surface_threat(obs)
    comp = agent._compound_threat(obs)
    enem = agent._threat(obs)
    obs, r, term, trunc, info = env.step(act)
    abs_px = SPAWN_ABS_PX + obs[0] * 512.0
    if abs_px > 2550:
        print(f"step {step:3d} | act {act} (surf={surf}, comp={comp}, enem={enem}) | x={abs_px:.1f} y={obs[1]*512:.1f} vy={obs[3]:.2f} g={obs[4]:.0f}")
    if term:
        break

env.close()
