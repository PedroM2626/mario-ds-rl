"""Stage 3: deploy the PINN-world-model agent in REAL TIME on the emulated
New Super Mario Bros. DS console and measure whether it clears the level.

Two controllers (both consume the *same* RAM observation, zero-shot):
  * mpc : closed-loop CEM-MPC planning inside the hard-residual PINN (no policy
          training; re-plans every executed step from the true state).
  * ppo : amortised Dyna-PPO policy trained inside imagination (<1 ms/frame).

Rendering streams the emulator's top screen through an OpenCV window so the run
is directly watchable; use --headless for batched metric collection.

Usage
-----
    # watch the MPC agent play live
    python src/eval_pinn_realtime.py --controller mpc --episodes 1 --render
    # headless, 5 episodes, metrics to evals/pinn_realtime_eval.json
    python src/eval_pinn_realtime.py --controller ppo --episodes 5

Output
------
    evals/pinn_realtime_eval.json
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ram_env import MarioRamEnv  # noqa: E402
from pinn_world_model import MarioPINNWorldModel, CEMMPController, N_ACTIONS, ShapingConfig  # noqa: E402
from course import Course, TILE  # noqa: E402
from tilemap_planner import TilemapAStar  # noqa: E402

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
STATE = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"
SPAWN_ABS_PX = 88.0  # Mario object B at entrance (see src/course.py calibration)
FRAMESKIP = 8
FRAME_DT = FRAMESKIP / 60.0  # wall-clock a single env step represents (s)


@contextlib.contextmanager
def _silence_c_stdout(enabled=True):
    if not enabled:
        yield
        return
    try:
        out_fd = sys.stdout.fileno()
        saved = os.dup(out_fd)
        devnull = os.open(os.devnull, os.WRONLY)
        sys.stdout.flush(); os.dup2(devnull, out_fd); os.close(devnull)
        yield
        sys.stdout.flush(); os.dup2(saved, out_fd); os.close(saved)
    except Exception:
        yield


def goal_px():
    """Absolute pixel X of the level exit (last solid ground column of World 1-1)."""
    try:
        c = Course(rom_path=ROM)
        return max(c.ground_cover.keys()) * TILE
    except Exception:
        return 4256


class MPCAgent:
    name = "mpc"

    def __init__(self, model, device, horizon, n_cand, n_iter, seed, cfg=None):
        self.ctrl = CEMMPController(model, horizon=horizon, n_candidates=n_cand,
                                    n_iters=n_iter, device=device, seed=seed, cfg=cfg)
        self._prev = None

    def act(self, obs):
        a, self._prev = self.ctrl.plan(obs, best_prev=self._prev)
        return a

    def reset(self):
        self._prev = None


class ReflexiveMPCAgent:
    """Reflex layer + model-based planner (reference S10.33 'Reflexive MPC').

    The learned world model cannot foresee dynamic Goomba contact, so pure MPC
    stalls on the first enemy. We compute an *exact* hazard reflex from the true
    egocentric RAM state (upcoming pits via geometry flags, and enemies via their
    live relative offset): when a hazard is close and Mario is grounded, commit a
    running jump; otherwise defer to the MPC planner for speed/timing decisions.
    """

    name = "mpc+reflex"
    JUMP = 4   # right + dash + jump  (running leap: clears pits, pipes, tall walls)
    HOP = 2    # right + jump         (short walk-hop: climbs 1-tile staircase steps)
    RUN = 3    # right + dash         (maintain running speed)
    LEFT = 5   # used briefly to gain runway when deeply stuck
    STUCK_ACTIONS = (4, 2, 6, 5)  # escape repertoire: run-jump, walk-jump, dash+jump, jump

    def __init__(self, model, device, horizon, n_cand, n_iter, seed, cfg=None,
                 enemy_lo=8.0, enemy_hi=42.0, enemy_dy=60.0, pit_lookahead=3, enabled=True,
                 pit_enemy_suppress=0.0, planner=None):
        self.mpc = MPCAgent(model, device, horizon, n_cand, n_iter, seed, cfg)
        self.enabled = enabled
        self.enemy_lo, self.enemy_hi, self.enemy_dy = enemy_lo, enemy_hi, enemy_dy
        self.pit_lookahead = pit_lookahead
        self.pit_enemy_suppress = pit_enemy_suppress
        self.planner = planner     # TilemapAStar: whole-level precise takeoff, else obs flags
        self.clearing = False   # mid jump-over-hazard: carry momentum through
        self._last_x = None     # for generic "blocked by wall/ledge" detection
        self._stuck = 0
        self._escape = 0
        self._retreat = 0
        self._blocked = 0
        self._queue = []        # scripted unstick maneuver (retreat -> charge -> leap)

    def _surface_threat(self, obs):
        """Geometry decision. Pits use the authoritative RAM ground flags (proven);
        tall pipes/walls (a >=3-tile rise in the ROM surface profile) get a preemptive
        running leap. Smaller steps are left to the blocked-jump: leaping them early
        throws Mario into nearby Goombas (empirically worse)."""
        if self.pit_lookahead > 0 and any(obs[17 + c] < 0.5 for c in range(self.pit_lookahead)):
            return "leap"                                  # pit ahead -> running leap
        if self.planner is not None:
            abs_px = SPAWN_ABS_PX + obs[0] * 512.0
            cur, prof = self.planner.surface_ahead(abs_px, look_tiles=3)
            if cur is None:
                return None
            for _c, row in prof:
                if row is None:
                    continue                               # pits handled by flags above
                if cur - row >= 3:                         # pipe / tall wall -> leap
                    return "leap"
        return None

    def _threat(self, obs):
        ene = 1e9
        for k in range(3):
            dx = obs[8 + 3 * k] * 256.0
            dy = obs[9 + 3 * k] * 256.0
            et = obs[10 + 3 * k] * 256.0
            if et > 1.0 and self.enemy_lo < dx < self.enemy_hi and abs(dy) < self.enemy_dy:
                return "enemy"
            if et > 1.0 and abs(dy) < self.enemy_dy and 0 < dx:
                ene = min(ene, dx)
        return None

    def act(self, obs):
        on_ground = obs[4] > 0.5
        vy = obs[3]
        # generic blocked detection: driving right but not advancing => wall/ledge
        if self._last_x is not None and on_ground:
            self._stuck = self._stuck + 1 if obs[0] - self._last_x < 0.0004 else 0
        else:
            self._stuck = 0
        self._last_x = obs[0]
        if self._queue:                         # play out a scripted unstick maneuver
            return self._queue.pop(0)
        if self.enabled and self.clearing:
            if not on_ground:
                return self.RUN
            self.clearing = False
        blocked = self._stuck >= 3
        enemy = self._threat(obs) if self.enabled else None
        surf = self._surface_threat(obs) if self.enabled else None
        if self.enabled and on_ground and vy > -0.05:
            if blocked:
                self._stuck = 0
                self._blocked += 1
                if self._blocked >= 3:
                    # arrived already stopped against a wall: a standstill leap goes
                    # straight up (no forward momentum). Back off, then charge a real
                    # running leap over it.
                    self._blocked = 0
                    self._queue = [self.LEFT] * 4 + [self.RUN] * 8 + [self.JUMP]
                else:
                    return self.JUMP
            elif enemy is not None or surf is not None:
                self.clearing = True
                return self.HOP if surf == "hop" else self.JUMP
            else:
                self._blocked = 0
        return self.mpc.act(obs)

    def reset(self):
        self.clearing = False
        self._last_x = None
        self._stuck = 0
        self._escape = 0
        self._retreat = 0
        self._blocked = 0
        self._queue = []
        self.mpc.reset()


class ReactiveAgent(ReflexiveMPCAgent):
    """Ablation: hazard reflex + continuous rightward drive, no MPC.

    Establishes the practical ceiling of the exact-state reflex alone given the
    6-action walking-only control set (no dash), i.e. how far geometry+timing
    heuristics get without the learned world model planning.
    """

    name = "reactive"

    def __init__(self, *a, **k):
        # no world-model planner needed; give it a dummy MPC that just runs right
        class _Run:
            def act(self, obs):
                return ReactiveAgent.RUN
            def reset(self):
                pass
        self.mpc = _Run()
        self.enabled = k.pop("enabled", True)
        self.enemy_lo = k.pop("enemy_lo", 8.0)
        self.enemy_hi = k.pop("enemy_hi", 42.0)
        self.enemy_dy = k.pop("enemy_dy", 60.0)
        self.pit_lookahead = k.pop("pit_lookahead", 3)
        self.pit_enemy_suppress = k.pop("pit_enemy_suppress", 150.0)
        self.planner = k.pop("planner", None)
        self.clearing = False
        self._last_x = None
        self._stuck = 0
        self._escape = 0
        self._retreat = 0
        self._blocked = 0
        self._queue = []


class PPOAgent:
    name = "ppo"

    def __init__(self, policy_path, device):
        from stable_baselines3 import PPO
        self.model = PPO.load(policy_path, device=device)

    def act(self, obs):
        a, _ = self.model.predict(obs[None], deterministic=True)
        return int(a[0])

    def reset(self):
        pass


class ReflexivePPOAgent(PPOAgent):
    """Hybrid PPO + safety reflex to prevent deterministic deaths at flat-ground enemies."""
    name = "ppo+reflex"

    def __init__(self, policy_path, device, planner=None,
                 enemy_lo=8.0, enemy_hi=46.0, enemy_dy=50.0):
        super().__init__(policy_path, device)
        self.planner = planner
        self.enemy_lo, self.enemy_hi, self.enemy_dy = enemy_lo, enemy_hi, enemy_dy
        self.clearing = False

    def act(self, obs):
        on_ground = obs[4] > 0.5
        vy = obs[3]
        if self.clearing:
            if not on_ground:
                return 3  # Maintain right + dash speed during jump
            self.clearing = False

        if on_ground and vy > -0.05:
            # Check dynamic enemy proximity: preemptive running leap over Goombas
            for k in range(3):
                dx = obs[8 + 3 * k] * 256.0
                dy = obs[9 + 3 * k] * 256.0
                et = obs[10 + 3 * k] * 256.0
                if et > 1.0 and self.enemy_lo < dx < self.enemy_hi and abs(dy) < self.enemy_dy:
                    self.clearing = True
                    return 4  # Action 4: right + dash + jump (running leap)
            # Check pit hazard ahead
            if any(obs[17 + c] < 0.5 for c in range(2)):
                self.clearing = True
                return 4

        return super().act(obs)

    def reset(self):
        self.clearing = False


def grab_top_screen(emu):
    buf = np.array(emu.display_buffer_as_rgbx(), dtype=np.uint8).reshape(384, 256, 4)
    return buf[:192, :, :3][:, :, ::-1].copy()  # top screen, BGR


def render_frame(emu, label, speed, writer=None):
    try:
        import cv2
        top = grab_top_screen(emu)
        top = cv2.resize(top, (256 * 2, 192 * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(top, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if writer is not None:
            writer.write(top)
        cv2.imshow("PINN NSMB-DS Agent (real-time)", top)
        cv2.waitKey(max(1, int(1000 * FRAME_DT / speed)))
    except Exception as e:  # pragma: no cover
        print("render disabled:", e)


def run_episode(env, agent, goal, render, speed, max_steps, verbose, writer=None):
    obs, _ = env.reset()
    agent.reset()
    total_r, steps = 0.0, 0
    max_rel_px = 0.0
    latencies = []
    cause, finished = "step_limit", False
    death_diag = ""
    t_wall = time.time()
    with _silence_c_stdout(enabled=not render):
        for steps in range(1, max_steps + 1):
            t0 = time.time()
            action = agent.act(obs)
            latencies.append((time.time() - t0) * 1000.0)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            max_rel_px = max(max_rel_px, obs[0] * 512.0)
            abs_px = SPAWN_ABS_PX + obs[0] * 512.0
            if render or writer is not None:
                render_frame(env.emu, f"step {steps} | x={abs_px:.0f}px | goal {goal}px",
                             speed, writer)
            if abs_px >= goal:
                finished, cause = True, "goal"
                break
            if term:
                cause = "death" if obs[6] > 0.02 else "timeout"
                if cause == "death":
                    ex = [round(float(obs[8 + 3 * k]) * 256) for k in range(3)]
                    et = [round(float(obs[10 + 3 * k]) * 256) for k in range(3)]
                    death_diag = (f"abs {SPAWN_ABS_PX + obs[0]*512:.0f}px vy={obs[3]:.2f} "
                                  f"ground={obs[4]:.0f} pit={[int(x) for x in obs[17:23]]} "
                                  f"enemy_dx_px={ex} enemy_type={et}")
                break
            if trunc:
                cause = "trunc"
                break
    wall = time.time() - t_wall
    return {
        "steps": steps, "total_reward": float(total_r),
        "max_progress_px": float(max_rel_px), "final_abs_px": float(SPAWN_ABS_PX + max_rel_px),
        "reached_goal": bool(finished), "termination": cause,
        "mean_control_latency_ms": float(np.mean(latencies)),
        "achieved_env_fps": float(steps / max(wall, 1e-9)),
        "death_diag": death_diag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["mpc", "ppo", "ppo+reflex", "reactive"], default="mpc")
    ap.add_argument("--model", default="models/pinn_nsmb.pt")
    ap.add_argument("--policy", default="models/pinn_policy.zip")
    ap.add_argument("--rom", default=ROM); ap.add_argument("--state", default=STATE)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n-cand", type=int, default=160)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--speed", type=float, default=2.0, help="wall-clock realtime multiplier when rendering")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", type=str, default="", help="write gameplay .avi to this path")
    ap.add_argument("--record-episode", type=int, default=1, help="which episode (1-based) to record")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--goal-px", type=int, default=0, help="override goal abs px (0=auto from ROM)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--w-progress", type=float, default=2.0)
    ap.add_argument("--w-enemy", type=float, default=0.0)
    ap.add_argument("--w-pit", type=float, default=0.0)
    ap.add_argument("--no-reflex", action="store_true", help="disable hazard reflex")
    ap.add_argument("--enemy-lo", type=float, default=8.0)
    ap.add_argument("--enemy-hi", type=float, default=55.0)
    ap.add_argument("--enemy-dy", type=float, default=60.0)
    ap.add_argument("--pit-lookahead", type=int, default=1, help="jump at the pit edge (A*-like takeoff)")
    ap.add_argument("--w-uncertainty", type=float, default=0.6,
                    help="pessimism on unpredictable enemy motion (probabilistic head)")
    ap.add_argument("--pit-enemy-suppress", type=float, default=0.0,
                    help="skip pit-jump if an enemy is within this many px (0=always jump)")
    args = ap.parse_args()

    render = args.render and not args.headless
    device = args.device
    goal = args.goal_px or goal_px()
    planner = TilemapAStar(rom_path=args.rom)   # ROM geometry -> wall/pipe perception
    print(f"[pinn-eval] controller={args.controller} device={device} goal_abs_px={goal} "
          f"wall_planner=on")

    if args.controller == "mpc":
        ck = torch.load(args.model, map_location=device, weights_only=False)
        model = MarioPINNWorldModel(hid=ck["hid"]).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
        cfg = ShapingConfig(progress=args.w_progress, enemy=args.w_enemy, pit=args.w_pit,
                            uncertainty=args.w_uncertainty)
        make_agent = lambda: ReflexiveMPCAgent(model, device, args.horizon, args.n_cand,
                                               args.n_iter, args.seed, cfg,
                                               enemy_lo=args.enemy_lo, enemy_hi=args.enemy_hi,
                                               enemy_dy=args.enemy_dy, pit_lookahead=args.pit_lookahead,
                                               pit_enemy_suppress=args.pit_enemy_suppress,
                                               planner=planner, enabled=not args.no_reflex)
    elif args.controller == "reactive":
        make_agent = lambda: ReactiveAgent(
            None, device, args.horizon, args.n_cand, args.n_iter, args.seed,
            enemy_lo=args.enemy_lo, enemy_hi=args.enemy_hi, enemy_dy=args.enemy_dy,
            pit_lookahead=args.pit_lookahead, pit_enemy_suppress=args.pit_enemy_suppress,
            planner=planner, enabled=not args.no_reflex)
    elif args.controller == "ppo+reflex":
        make_agent = lambda: ReflexivePPOAgent(
            args.policy, device, planner=planner,
            enemy_lo=args.enemy_lo, enemy_hi=args.enemy_hi, enemy_dy=args.enemy_dy)
    else:
        make_agent = lambda: PPOAgent(args.policy, device)

    env = MarioRamEnv(rom_path=args.rom, state_path=args.state, geo=True,
                      max_steps=args.max_steps)
    writer = None
    if args.record:
        import cv2
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"MJPG"), 30, (512, 384))
    results = []
    for ep in range(args.episodes):
        agent = make_agent()
        w = writer if (writer is not None and ep + 1 == args.record_episode) else None
        res = run_episode(env, agent, goal, render, args.speed, args.max_steps, verbose=True, writer=w)
        res["episode"] = ep + 1
        results.append(res)
        flag = "LEVEL CLEARED" if res["reached_goal"] else "did not finish"
        print(f"[pinn-eval] ep {ep+1}: progress {res['final_abs_px']:.0f}px / {goal}px "
              f"({100*res['final_abs_px']/goal:.0f}%) | steps {res['steps']} | "
              f"reward {res['total_reward']:.1f} | {res['termination']} | "
              f"{res['achieved_env_fps']:.1f} fps | {res['mean_control_latency_ms']:.1f} ms/act -> {flag}")
        if res.get("death_diag"):
            print(f"           death: {res['death_diag']}")
    env.close()
    if writer is not None:
        writer.release()
        print(f"[pinn-eval] recorded -> {args.record}")
    if render:
        import cv2; cv2.destroyAllWindows()

    agg = {
        "controller": args.controller, "episodes": args.episodes, "goal_abs_px": goal,
        "clear_rate": float(np.mean([r["reached_goal"] for r in results])),
        "mean_progress_px": float(np.mean([r["final_abs_px"] for r in results])),
        "max_progress_px": float(np.max([r["final_abs_px"] for r in results])),
        "mean_reward": float(np.mean([r["total_reward"] for r in results])),
        "mean_fps": float(np.mean([r["achieved_env_fps"] for r in results])),
        "mean_latency_ms": float(np.mean([r["mean_control_latency_ms"] for r in results])),
    }
    os.makedirs("evals", exist_ok=True)
    out = {"aggregate": agg, "episodes": results}
    with open("evals/pinn_realtime_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[pinn-eval] aggregate:", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
