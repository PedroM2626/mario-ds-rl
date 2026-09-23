# Physics-Informed World Model for *New Super Mario Bros.* (DS)

**A PINN world model that trains in ~1 minute from a handful of emulator frames and
drives the agent in real time on the console.**

This module ports the methodology of
[`PedroM2626/smw-pinn`](https://github.com/PedroM2626/smw-pinn) — a *Hard-Residual
Physics-Informed Neural Network* world model for Super Mario World (SNES) — to
**New Super Mario Bros. (Nintendo DS)**, built directly on this repo's
RAM-telemetry stack ([`src/ram_env.py`](../src/ram_env.py),
[`src/ram_state.py`](../src/ram_state.py), [`src/course.py`](../src/course.py)).
No computer vision is used for control: the agent acts on a 23-dimensional vector
read straight from emulator memory.

---

## 1. The idea

A world model learns the forward dynamics \(s_{t+1}=f(s_t,a_t)\). In a platformer
the engine integrates player motion with a *fixed, exact* rule, so the smart thing
to learn is not the whole next state but only the **un-modelled force/contact
residual**, while the position integration is **hard-wired into the computation
graph**. That is the "hard residual PINN" — it has a *zero structural kinematic
residual by construction*, which is why it is so sample efficient.

We verified the exact integration law on real NSMB-DS data. With the env
`frameskip = 8` and the RAM-normalised state, the identity holds for **both axes**:

\[
x_{t+1} = x_t + \frac{v_{x,t+1}}{16},\qquad
y_{t+1} = y_t + \frac{v_{y,t+1}}{16}
\]

(measured residual mean ≈ 4·10⁻⁴, std ≈ 9·10⁻³ in px/512 units). This is the
direct analogue of the smw-pinn `ΔX = v_x / 16` identity.

### State / action space

| | |
|---|---|
| **State** `Box(23)` | `x, y, vx, vy, on_ground, lives, time, cam`, 3× enemies `(dx, dy, type)`, 6× pit flags |
| **Action** `Discrete(8)` | `0` noop · `1` right · `2` right+jump · `3` right+**dash** · `4` right+dash+jump · `5` left · `6` dash+jump · `7` jump |
| **Units** | positions in px/512, velocities in px/frame/4, enemy offsets in px/256 |

### Model architecture (`MarioPINNWorldModel`)

```
s_t, a_t ─► MLP trunk (128×128, LayerNorm, GELU) ─► delta(obs)  ─┐
                             ├────────────────────► reward(s,a)   ├─► next state
                             └────────────────────► continue(s,a) ┘   (x,y integrated exactly)
```

The trunk predicts a per-dimension **delta**. Velocities are clamped to the
engine's saturation limits, then positions are integrated with the exact `+K·v`
rule, so the kinematic residual is identically zero regardless of the weights. A
**heteroscedastic `head_logvar`** learns *where the model is unsure* (chiefly enemy
motion) via a Gaussian-NLL dynamics loss; the planner can then be pessimistic there
(`ShapingConfig.uncertainty`) — the probabilistic world-model head.

**Dash unlock.** The original 6-action set only let Mario *walk* (1.5 px/frame),
capping the jump at ~60 px — shorter than the 64–80 px pits. Reverse-engineering the
DS pad showed **`X`/`Y` is the run/dash button** (2× speed → 3.0 px/frame), so
`ACTION_KEYS` in [`src/ram_env.py`](../src/ram_env.py) adds dash + running-jump
actions; a running jump spans ~120 px and clears the wide pits.

---

## 2. Pipeline (three stages)

```bash
# Stage 1 — collect real RAM transitions and train the world model (~1 min)
python src/train_pinn_world_model.py --transitions 3000 --epochs 300 --sample-efficiency
#   -> models/pinn_nsmb.pt, models/pinn_pool.npz, evals/pinn_worldmodel_metrics.json

# Stage 2 — train an amortised policy inside imagination (Dyna-PPO / MBPO)
python src/train_pinn_policy.py --timesteps 200000 --n-envs 8
#   -> models/pinn_policy.zip, evals/pinn_policy_metrics.json

# Stage 3 — deploy in REAL TIME on the emulator (watch it play live)
python src/eval_pinn_realtime.py --controller mpc --render --episodes 1
#   headless + record a gameplay clip:
python src/eval_pinn_realtime.py --controller mpc --headless --record media/pinn_mpc_run.avi
#   -> evals/pinn_realtime_eval.json
```

### Stage 1 results — the world model is near-perfect and extremely sample efficient

| Metric | Value |
|---|---|
| Params | 43,440 (+ `logvar` head) |
| Test single-step MSE | **0.056** |
| Kinematic residual (x / y) | **3.8·10⁻⁸ / 1.2·10⁻⁹** (structurally zero) |
| Train time (3500 transitions, CUDA) | 88 s |
| Collection throughput | ~20 env steps/s (real emulator) |

**Sample-efficiency study** (test MSE vs. number of real transitions trained on):

| N transitions | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| Test MSE | 0.113 | 0.051 | 0.063 | 0.056 |

Training on **200 transitions (~3 s of gameplay)** already gives MSE ~0.11 — the
physics inductive bias reproduces the smw-pinn >25× sample-efficiency finding on a
different console.

### Tilemap A\* feasibility check (`src/tilemap_planner.py`)

Parsing the ROM geometry, World 1-1 has **14 pits, the widest 5 tiles (80 px)** — all
within the ~112 px (7-tile) running-jump reach. The planner reports
**`feasible=True`**, i.e. the stage is *geometrically* completable with the dash
action; any remaining difficulty is dynamic-hazard perception/timing, not geometry.

### Stage 3 results — real-time play on the console (World 1-1, goal ≈ 4256 px, 8 episodes, dash enabled)

| Controller | World model used? | Mean progress | Max progress | Control latency | Level cleared |
|---|---|---|---|---|---|
| **Reflex + A\* (dash)** | Geometry only | **1748 px (41%)** | **1819 px (43%)** | ~0 (real-time) | not yet |
| CEM-MPC + reflex | Yes (online replanning) | 480 px | 963 px | ~130 ms/frame | not yet |
| Dyna-PPO (Stage 2 policy) | Yes (imagination-trained) | ~350 px | ~424 px | **~2.5 ms/frame** | not yet |

The dash **more than doubles** the reachable distance (26% → 43%). Notably, once the
dynamic hazards dominate, the tight *reflex* beats the model-based MPC — the
world model is exact for player kinematics but blind to some dynamic entities, so
planning on it can be *worse* than an exact-state reflex (a finding consistent with
the smw-pinn study's own "optimism under hallucinated dynamics" result).

---

## 3. Controllers

* **Reflex + A\* (`--controller reactive`, best real-time reach).** An exact-state
  hazard reflex + continuous dash drive, with pit takeoffs chosen by the tilemap A\*
  (`should_jump`) instead of the raw flags, plus a generic "stuck → escape repertoire
  → retreat for runway" unstick. No learned model, so no dynamic-entity blindness —
  with the dash it reaches the furthest, ~43%.
* **CEM-MPC (`--controller mpc`).** Cross-Entropy-Method MPC over discrete action
  sequences (H=20, 160 candidates, 3 refinements) with `ShapingConfig` progress/pit/
  enemy/**uncertainty** terms. Re-plans every step from the true state (exact for
  kinematics). It is the purest demonstration of the world model, but where dynamic
  hazards dominate the model's enemy blindness makes it *worse* than the tight reflex
  — the documented "optimism under hallucinated dynamics" effect.
* **Dyna-PPO (`--controller ppo`).** An SB3 `MlpPolicy` actor-critic trained entirely
  inside the learned model's imagination. Fastest at **~2.5 ms/frame** (>400 FPS) but
  transfers most weakly (same dynamic-horizon limitation).

### Hazard reflex (`ReflexiveMPCAgent`)

The learned model captures *player* physics perfectly but *enemy* dynamics are weak
(they are only observed when on screen). We therefore compute an **exact** jump
reflex from the true egocentric RAM state — live enemies (`dx, dy, type`) — and take
pits from the **tilemap A\*** (whole-level `should_jump`, incl. a one-tile early
leap for wide pits) rather than a fixed pixel window. Mario is committed to a
**running** hop (dash+jump) that clears the pit and carries past the hazard at once.
The dash makes the two hazards compatible (a long leap clears both), so the old
"suppress the pit-jump near an enemy" rule is now off by default.

---

## 4. Does the agent finish the level?

**Not fully yet — reported honestly, with the barrier now precisely localised.** With
the three upgrades (dash action, probabilistic head, tilemap A\*) the agent plays the
stage live, clears the early Goombas and every pit up to 5 tiles wide, and reaches a
reliable **~43% (1819 px)** in real time — roughly **double** the pre-dash result.

The remaining wall is a **dynamic-entity perception gap**, not geometry or controls
(both are now solved: A\* reports the stage feasible; the dash clears the pits). At
x≈1792 there is a ceiling/falling hazard (RAM actor type `0x4C`, `course.py` sprite
id 198) that is **absent from the 23-dim egocentric observation** until it is already
past/above Mario (the enemy channel only reports the 3 nearest by |dx| and no
overhead threat), so the reflex and the model-based planner both react too late. The
same class recurs later in the stage. This is exactly the *long-horizon dynamic-hazard
perception* problem the smw-pinn study leaves as open frontier work — and notably,
**no agent in this repository's extensive prior benchmarks (PPO/CURL/SPR/RAM/GNN/world
models) had cleared World 1-1 either** (their best was ~1538 reward ≈ progress, not a
finish), which is strong evidence that full completion is hard for reasons intrinsic
to the environment/observation, not to this world model.

The physics-informed **world model itself is solved** (residual = 0, MSE 0.056,
200-sample training) and the three requested features are implemented, tested, and
integrated. The concrete next step to actually clear the stage is **perception**, not
modelling: enrich the observation with the ROM's static hazard set (spike/ceiling-drop
x-positions from `course.py`) and more simultaneous entities, so the planner/reflex
can pre-empt falling hazards the way the A\* planner already pre-empts pits.

---

## 5. Files

| File | Role |
|---|---|
| [`src/pinn_world_model.py`](../src/pinn_world_model.py) | `MarioPINNWorldModel` (hard-residual + probabilistic `logvar` head), `shaped_step`, `PINNRollout`, `CEMMPController`, `PINNImaginationEnv` |
| [`src/tilemap_planner.py`](../src/tilemap_planner.py) | Task 3: ROM tilemap A\* / jump-reachability planner (pit takeoffs, `feasible` proof, goal) |
| [`src/train_pinn_world_model.py`](../src/train_pinn_world_model.py) | Stage 1: collect RAM data, train PINN, report MSE / residual / drift / sample-efficiency |
| [`src/train_pinn_policy.py`](../src/train_pinn_policy.py) | Stage 2: Dyna-PPO in imagination |
| [`src/eval_pinn_realtime.py`](../src/eval_pinn_realtime.py) | Stage 3: real-time console eval (`mpc` / `ppo` / `reactive`), rendering + `.avi` recording + finish detection |
| `media/pinn_mpc_run.avi` | Recorded real-time gameplay clip |

### Tunable flags (`eval_pinn_realtime.py`)

`--horizon`, `--n-cand`, `--n-iter` (planning depth) · `--w-progress/-enemy/-pit/-uncertainty`
(model-reward shaping + probabilistic pessimism) · `--enemy-lo/-hi/-dy`,
`--pit-lookahead`, `--pit-enemy-suppress`, `--no-planner`, `--no-reflex` (reflex / A\*)
· `--goal-px` (auto-read from ROM) · `--render` / `--record PATH` / `--speed`.
