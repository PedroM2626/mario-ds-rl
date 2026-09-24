import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))
from src.ram_env import MarioRamEnv
from src.eval_pinn_realtime import ReactiveAgent, SPAWN_ABS_PX, TilemapAStar

planner = TilemapAStar()
agent = ReactiveAgent(planner=planner)
env = MarioRamEnv(geo=True, max_steps=175)
obs, _ = env.reset()

print("Tracing steps 150-170 (2650px - 2850px):")
for step in range(1, 175):
    act = agent.act(obs)
    obs, r, term, trunc, info = env.step(act)
    abs_px = SPAWN_ABS_PX + obs[0] * 512.0
    if abs_px > 2650:
        ens = info.get("enemies", [])
        en_desc = [(e.get("name"), round(e.get("dx", 0)), round(e.get("dy", 0)), e.get("type")) for e in ens[:3]]
        lives = obs[5] * 10.0
        time_left = obs[6] * 400.0
        print(f"step {step:3d} | act {act} | x={abs_px:6.1f} y={obs[1]*512:5.1f} vy={obs[3]:5.2f} g={obs[4]:.0f} lives={lives:.0f} time={time_left:.0f} | clearing={agent.clearing} | blocked={agent._blocked} stuck={agent._stuck} queue={agent._queue} | enemies: {en_desc}")
    if term:
        print(f"Terminated at step {step}: abs_x={abs_px:.1f}, term={term}, trunc={trunc}, info={info}")
        break

env.close()
