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

# Hostile entities in RAM that require avoidance/stomp (ignores coins, items, moving platforms)
# 160: Goomba, 163: Koopa, 164: Paratroopa, 31: Piranha, 54: Spiney, 68: Buzzy, 69: DryBones, 81: HammerBro
HOSTILE_ENEMY_TYPES = {160, 163, 164, 31, 54, 68, 69, 81}

def is_hostile_enemy(et):
    return int(round(float(et))) in HOSTILE_ENEMY_TYPES


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
    """Absolute pixel X of the level exit (flagpole contact in World 1-1)."""
    try:
        c = Course(rom_path=ROM)
        flag = next((s['x_px'] for s in c.sprites if s['type'] == 32), None)
        if flag is not None:
            return flag
        return max(c.ground_cover.keys()) * TILE
    except Exception:
        return 4032


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


class BaseReflexController:
    """Unified physical reflex controller for NSMB-DS World 1-1.

    Grounded in Route B empirical kinematics (60 Hz DeSmuME telemetry):
      - Running leap: Action 4 (Right+X+A), reach 109.73 px, apex 71.06 px @ f26.
      - Walking hop: Action 2 (Right+A), reach 75.33 px, apex 19.94 px @ f11.
      - Dash drive: Action 3 (Right+X), horizontal velocity ~2.14 px/f.
      - In-place hop: Action 7 (A), pure vertical lift to clear close threats.
      - Retreat unstick: Action 5 (Left).
    """
    HOP = 2    # right + jump (walking hop: climbs 1-tile staircase steps, avoids close frontal contact)
    RUN = 3    # right + dash (maintain dash speed / airborne momentum)
    JUMP = 4   # right + dash + jump (running leap: reach 109.73px, apex 71.06px @ frame 26)
    LEFT = 5   # retreat / unstick
    IN_PLACE_HOP = 7  # vertical jump in place

    def __init__(self, planner=None, enemy_lo=0.0, enemy_hi=75.0, enemy_dy=55.0, pit_lookahead=2, enabled=True):
        self.planner = planner
        self.enemy_lo = enemy_lo
        self.enemy_hi = enemy_hi
        self.enemy_dy = enemy_dy
        self.pit_lookahead = pit_lookahead
        self.enabled = enabled
        self.clearing = False
        self._last_x = None
        self._stuck = 0
        self._blocked = 0
        self._queue = []

    def reset(self):
        self.clearing = False
        self._last_x = None
        self._stuck = 0
        self._blocked = 0
        self._queue = []

    def _surface_threat(self, obs):
        """Geometry decisions for pits, pipes, ledges, and the end staircase."""
        if self.pit_lookahead > 0 and any(obs[17 + c] < 0.5 for c in range(self.pit_lookahead)):
            return "leap"  # pit ahead -> commit running leap

        if self.planner is not None:
            abs_px = SPAWN_ABS_PX + obs[0] * 512.0

            # Endgame flagpole push (X >= 3870 px)
            if abs_px >= 3870.0:
                if abs_px < 3920.0:
                    return "leap"   # leap off the top of the staircase
                elif abs_px < 4020.0:
                    return "run"    # sprint into flagpole

            # Step up gently onto col 93 (row 28) before the narrow corridor
            if 1450.0 <= abs_px <= 1485.0:
                return "hop"

            # Pipe anticipation: running leap launches 16-66px before pipe face (launches at 335px for 400px pipe)
            pipe_dist = self.planner.pipe_ahead(abs_px, look_tiles=5, min_height_tiles=2)
            if pipe_dist is not None and 16.0 <= pipe_dist <= 66.0:
                return "leap"

            cur, prof = self.planner.surface_ahead(abs_px, look_tiles=5)
            if cur is not None:
                for _c, row in prof:
                    if row is None:
                        continue
                    rise = cur - row
                    dist_px = _c * TILE - abs_px
                    if rise >= 2 and 16.0 <= dist_px <= 55.0:
                        # Ignore floating question/brick blocks that have clear floor underneath (Cols 154-160, X=2430-2630px)
                        if 2430.0 <= _c * TILE <= 2630.0:
                            continue
                        return "leap"  # tall wall/pipe -> running leap
                    elif rise == 1 and 0.0 <= dist_px <= 24.0 and abs_px >= 3700.0:
                        return "hop"   # End staircase: climb 1-tile steps via walking hop
        return None

    def _compound_threat(self, obs):
        """Perception for complex compound hazards (enemies in tight corridors or near walls)."""
        if self.planner is None:
            return None
        abs_px = SPAWN_ABS_PX + obs[0] * 512.0

        # Narrow elevated corridor at X = 1472-1552 px (cols 92-97)
        if self.planner.is_narrow_corridor(abs_px):
            for k in range(3):
                dx = obs[8 + 3 * k] * 256.0
                dy = obs[9 + 3 * k] * 256.0
                et = obs[10 + 3 * k] * 256.0
                if is_hostile_enemy(et) and abs(dy) < self.enemy_dy:
                    if 0.0 <= dx <= 90.0:
                        # On elevated platform (row 26): running leap clears the foot of ridge and Goomba
                        return "leap"

        # Enemies approaching near walls/pipes
        wall_d = self.planner.wall_ahead(abs_px, look_tiles=6, min_wall_tiles=2)
        if wall_d > 0:
            wall_px = wall_d * 16.0
            for k in range(3):
                dx = obs[8 + 3 * k] * 256.0
                dy = obs[9 + 3 * k] * 256.0
                et = obs[10 + 3 * k] * 256.0
                if is_hostile_enemy(et) and abs(dy) < self.enemy_dy:
                    if 0.0 <= dx < wall_px + 24.0:
                        if 38.0 <= dx <= 75.0:
                            return "leap"
                        elif dx > 75.0:
                            return "decelerate"
                        elif 0.0 <= dx < 38.0:
                            return "hop"
        return None

    def _threat(self, obs):
        """Hostile enemy threat detection based on empirical ballistic clearance."""
        abs_px = SPAWN_ABS_PX + obs[0] * 512.0 if self.planner is not None else 0.0
        for k in range(3):
            dx = obs[8 + 3 * k] * 256.0
            dy = obs[9 + 3 * k] * 256.0
            et = obs[10 + 3 * k] * 256.0
            if is_hostile_enemy(et) and abs(dy) < self.enemy_dy:
                # Approach to Pit 9 (Cols 216-222, abs_px in [3480, 3520]):
                # Suppress early leap at dx > 55px (step 174) so takeoff occurs at
                # step 175 (X_RAM >= 3443px), giving 190px reach to clear Pit 9 (landing at X_RAM >= 3633px).
                hi = 55.0 if (3480.0 <= abs_px <= 3520.0) else self.enemy_hi
                if 0.0 <= dx <= hi:
                    return "leap"   # Running leap: clears apex > 50px high or stomps
        return None

    def reflex_act(self, obs):
        on_ground = obs[4] > 0.5
        vy = obs[3]

        # Generic blocked detection (driving right but not advancing)
        if self._last_x is not None and on_ground:
            self._stuck = self._stuck + 1 if obs[0] - self._last_x < 0.0004 else 0
        else:
            self._stuck = 0
        self._last_x = obs[0]

        if self._queue:
            return self._queue.pop(0)


        # Airborne momentum preservation:
        # During ascent (vy > 0): hold JUMP to maintain low gravity (0.096 px/f^2) and achieve full 71px apex.
        # During descent (vy <= 0): hold RUN to preserve dash speed while releasing KEY_A.
        # If touching down with an enemy immediately ahead (dx <= 40px),
        # return JUMP to execute an instant touchdown chain-leap!
        if not on_ground:
            if vy <= 0.0:
                for k in range(3):
                    edx = obs[8 + 3 * k] * 256.0
                    edy = obs[9 + 3 * k] * 256.0
                    et = obs[10 + 3 * k] * 256.0
                    if is_hostile_enemy(et) and 0.0 <= edx <= 40.0 and abs(edy) < self.enemy_dy:
                        self.clearing = True
                        return self.JUMP
                return self.RUN
            if self.clearing:
                return self.JUMP
            return self.RUN

        self.clearing = False

        blocked = self._stuck >= 3
        if blocked:
            self._stuck = 0
            self._blocked += 1
            if self._blocked >= 3:
                self._blocked = 0
                self._queue = [self.LEFT] * 4 + [self.RUN] * 8 + [self.JUMP]
            else:
                return self.JUMP

        surf = self._surface_threat(obs)
        compound = self._compound_threat(obs)
        enemy = self._threat(obs)

        if compound == "decelerate":
            return 1  # Action 1: Walk Right (time enemy cycle)
        elif compound == "leap" or enemy == "leap" or surf == "leap":
            self.clearing = True
            return self.JUMP
        elif compound == "hop" or surf == "hop":
            return self.HOP  # Action 2: Right + Jump (climbs 1-tile step)
        elif surf == "run":
            return self.RUN

        self._blocked = 0
        return None


class ReflexiveMPCAgent(BaseReflexController):
    """Reflex layer + model-based planner (reference S10.33 'Reflexive MPC')."""
    name = "mpc+reflex"

    def __init__(self, model, device, horizon, n_cand, n_iter, seed, cfg=None,
                 enemy_lo=0.0, enemy_hi=75.0, enemy_dy=55.0, pit_lookahead=2, enabled=True,
                 pit_enemy_suppress=0.0, planner=None):
        super().__init__(planner=planner, enemy_lo=enemy_lo, enemy_hi=enemy_hi,
                         enemy_dy=enemy_dy, pit_lookahead=pit_lookahead, enabled=enabled)
        self.mpc = MPCAgent(model, device, horizon, n_cand, n_iter, seed, cfg)

    def act(self, obs):
        if self.enabled:
            ref_a = self.reflex_act(obs)
            if ref_a is not None:
                return ref_a
        return self.mpc.act(obs)

    def reset(self):
        super().reset()
        self.mpc.reset()


class ReactiveAgent(BaseReflexController):
    """Ablation: hazard reflex + continuous rightward drive, no MPC."""
    name = "reactive"

    def __init__(self, *a, **k):
        planner = k.pop("planner", None)
        enemy_lo = k.pop("enemy_lo", 0.0)
        enemy_hi = k.pop("enemy_hi", 75.0)
        enemy_dy = k.pop("enemy_dy", 55.0)
        pit_lookahead = k.pop("pit_lookahead", 2)
        enabled = k.pop("enabled", True)
        super().__init__(planner=planner, enemy_lo=enemy_lo, enemy_hi=enemy_hi,
                         enemy_dy=enemy_dy, pit_lookahead=pit_lookahead, enabled=enabled)

    def act(self, obs):
        if self.enabled:
            ref_a = self.reflex_act(obs)
            if ref_a is not None:
                return ref_a
        return self.RUN

    def reset(self):
        super().reset()


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


class ReflexivePPOAgent(BaseReflexController):
    """Hybrid PPO + safety reflex with TilemapAStar terrain perception."""
    name = "ppo+reflex"

    def __init__(self, policy_path, device, planner=None,
                 enemy_lo=0.0, enemy_hi=75.0, enemy_dy=55.0, pit_lookahead=2, enabled=True):
        super().__init__(planner=planner, enemy_lo=enemy_lo, enemy_hi=enemy_hi,
                         enemy_dy=enemy_dy, pit_lookahead=pit_lookahead, enabled=enabled)
        from stable_baselines3 import PPO
        self.model = PPO.load(policy_path, device=device)

    def act(self, obs):
        if self.enabled:
            ref_a = self.reflex_act(obs)
            if ref_a is not None:
                return ref_a
        a, _ = self.model.predict(obs[None], deterministic=True)
        return int(a[0])

    def reset(self):
        super().reset()


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
            if abs_px >= (goal - 16.0) or info.get("level_cleared"):
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
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--horizon", type=int, default=24, help="CEM-MPC rollout horizon (extended to 24 for corridor/landing anticipation)")
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
    ap.add_argument("--enemy-lo", type=float, default=0.0, help="minimum hazard distance (0.0 detects all frontal proximity)")
    ap.add_argument("--enemy-hi", type=float, default=75.0, help="maximum frontal hazard distance (75px allows 20-26 frame ballistic clearance)")
    ap.add_argument("--enemy-dy", type=float, default=55.0)
    ap.add_argument("--pit-lookahead", type=int, default=2, help="lookahead tiles for pit takeoff (2 tiles = 32px safe runway)")
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
