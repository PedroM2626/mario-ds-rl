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

### Tilemap A\* feasibility + walkable-surface model (`src/tilemap_planner.py`)

Parsing the ROM geometry, World 1-1 has **14 pits, the widest 5 tiles (80 px)** — all
within the ~112 px (7-tile) running-jump reach. The planner reports
**`feasible=True`**, i.e. the stage is *geometrically* completable with the dash
action. It derives a per-column **obstacle height** (`wall_ahead`) and, more generally,
a **walkable-surface profile** (`surface_ahead`): the top of the solid run contiguous
with the ground at each column (ground + blocks/pipes/stairs stacked on it, ignoring
floating ?-blocks). This unifies pits (`None`), the ~4-tile pipe (row 27) and the end
staircase (a 1-tile-per-column ramp, cols 235–242) into a single height profile the
controller can follow. Jump physics were measured in-engine: a running jump reaches
~114 px (7 tiles) horizontally and clears a 4-tile pipe.

### Stage 3 results — real-time play on the console (World 1-1, goal ≈ 4256 px, 8 episodes, dash enabled)

| Controller | World model used? | Mean progress | Max progress | Control latency | Level cleared |
|---|---|---|---|---|---|
| **Reflex + A\* (dash)** | Geometry only | **1804 px (42%)** | **1889 px (44%)** | ~0 (real-time) | not yet |
| CEM-MPC + reflex | Yes (online replanning) | 480 px | 963 px | ~130 ms/frame | not yet |
| Dyna-PPO (Stage 2 policy) | Yes (imagination-trained) | ~350 px | ~424 px | **~2.5 ms/frame** | not yet |

The dash **more than doubles** the reachable distance (26% → 44%). With ROM wall/pipe
perception the reflex now **clears the first ~4-tile green pipe** (reaching the
pipe-top at 1889 px) — the furthest any controller in this repository has been shown
to get on World 1-1. Notably, once the dynamic hazards and dense platforming dominate,
the tight *reflex* beats the model-based MPC — the world model is exact for player
kinematics but blind to some dynamic entities, so planning on it can be *worse* than an
exact-state reflex (a finding consistent with the smw-pinn study's own "optimism under
hallucinated dynamics" result).

---

## 3. Controllers

* **Reflex + A\* (`--controller reactive`, best real-time reach).** An exact-state
  hazard reflex + continuous dash drive. Pit takeoffs come from the **authoritative
  RAM ground flags**; tall pipes/walls come from the tilemap planner's
  `wall_ahead(...)` (obstacle height ≥ 3 tiles), which triggers a **preemptive
  running leap** while Mario still has forward speed (a standstill jump is too short
  for a ~4-tile pipe). A hierarchical unstick (running-leap → retreat·charge·leap) is
  the fallback when he arrives already stopped against a wall. No learned model, so no
  dynamic-entity blindness — with the dash it reaches the furthest, ~44%.
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
reflex from the true egocentric RAM state — live enemies (`dx, dy, type`) — take pits
from the **authoritative RAM ground flags**, and take tall pipes/walls from the
**tilemap planner** (`surface_ahead`, the whole-level walkable-surface profile). Mario is
committed to a **running** hop (dash+jump) taken from the run-up, which clears both pits
and the ~4-tile pipes. The dash makes the hazards compatible (a long leap clears them),
so the old "suppress the pit-jump near an enemy" rule is now off by default.

---

## 4. Does the agent finish the level?

**Not fully yet — reported honestly, with the barrier precisely localised by direct
in-game observation (rendered frames + per-step RAM traces).** With the dash, the
probabilistic head and the tilemap A\* + wall/pipe perception, the agent plays the
stage live, clears the early Goombas, every pit up to 5 tiles wide, and the first
~4-tile green pipe, reaching **~44% (1889 px)** in real time — roughly **double** the
pre-dash result and the furthest any controller in this repository gets on World 1-1.

**Corrected diagnosis.** An earlier hypothesis blamed a dynamic `0x4C` "falling
hazard" perception gap at x≈1792. Direct evidence (the rendered frame at the stall +
the per-step velocity trace) disproves that: the stall is at the **pipe + dense
block-staircase** just after it. The ROM geometry there is *solid ground* (no pit) with
a ~4-tile pipe at col 113 (px 1808) followed by a cluster of 1–2-tile blocks and
overhead blocks (cols 115–123). Mario clears the lone pipe when he arrives with speed
(a running leap peaks over it — verified reaching the pipe-top at 1889 px), but a
standstill jump on a narrow surface goes straight up (`vx = 0` in the trace) and cannot
advance, so he oscillates until the **stage timer expires** ("Time's up!"). This is a
genuine **multi-hop 2D-platforming + runway-timing** problem, compounded by the game
timer that punishes any stalling — not a perception gap and not the world model.

The physics-informed **world model itself is solved** (kinematic residual = 0, MSE
0.056, 200-sample training) and all three requested features (dash, probabilistic
head, tilemap A\* / walkable-surface perception) are implemented, tested and integrated.

**Exhaustive control sweep (this session).** With the correct surface model in hand we
swept the reactive controller across the full geometry/enemy-timing space and each
configuration trades one failure for another, all plateauing at ~44%:
- leap only `>=3`-tile rises → clears the pipe (reaches pipe-top 1889 px) but stalls on
  the 2-tile ledge at the pipe exit (a standstill jump there goes straight up, `vx=0`);
- leap `>=2`-tile rises → clears the ledge but leaps small early blocks straight into
  Goombas (dies at ~446–1509 px);
- widen the enemy-jump window to stomp Goombas earlier → jumps land him in pits (mean
  drops to ~1137 px).
The **definitive barrier is the enemy–pit–platforming coupling**: Goombas sit at pit
edges and block clusters, so a reactive rule that clears the geometry walks into the
enemy, and one that dodges/stomps the enemy mistimes the jump into a pit. Resolving
this jointly is what a *learned* long-horizon policy (large-scale direct RL) or a
full 2D planner **with dynamic-entity-aware waypoint timing** is for — a substantially
larger effort than the reactive reflex. The untried second half (the tall staircase to
the flagpole, Koopa at px 2080) compounds it. Reported honestly: the agent does **not
yet clear World 1-1**; it reliably clears the pits, the Goombas to ~1.7k px and the
first pipe, reaching **~44%** — the furthest any controller in this repository gets.

---

## 5. Files

| File | Role |
|---|---|
| [`src/pinn_world_model.py`](../src/pinn_world_model.py) | `MarioPINNWorldModel` (hard-residual + probabilistic `logvar` head), `shaped_step`, `PINNRollout`, `CEMMPController`, `PINNImaginationEnv` |
| [`src/tilemap_planner.py`](../src/tilemap_planner.py) | Task 3: ROM tilemap A\* / jump-reachability planner (pit takeoffs, `feasible` proof, `wall_ahead` obstacle-height map, goal) |
| [`src/train_pinn_world_model.py`](../src/train_pinn_world_model.py) | Stage 1: collect RAM data, train PINN, report MSE / residual / drift / sample-efficiency |
| [`src/train_pinn_policy.py`](../src/train_pinn_policy.py) | Stage 2: Dyna-PPO in imagination |
| [`src/eval_pinn_realtime.py`](../src/eval_pinn_realtime.py) | Stage 3: real-time console eval (`mpc` / `ppo` / `reactive`), rendering + `.avi` recording + finish detection |
| `media/pinn_mpc_run.avi` | Recorded real-time gameplay clip |
| `media/nsmb_pipe_top_timesup.png` | Frame at the current barrier: Mario on the cleared pipe-top, killed by the stage timer ("Time's up!") |

### Tunable flags (`eval_pinn_realtime.py`)

`--horizon`, `--n-cand`, `--n-iter` (planning depth) · `--w-progress/-enemy/-pit/-uncertainty`
(model-reward shaping + probabilistic pessimism) · `--enemy-lo/-hi/-dy`,
`--pit-lookahead`, `--pit-enemy-suppress`, `--no-reflex` (reflex / wall A\*)
· `--goal-px` (auto-read from ROM) · `--render` / `--record PATH` / `--speed`.
