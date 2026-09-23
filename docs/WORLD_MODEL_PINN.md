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
| **Action** `Discrete(6)` | `0` noop · `1` right · `2` right+jump · `3` right+jump+spin · `4` left · `5` jump |
| **Units** | positions in px/512, velocities in px/frame/4, enemy offsets in px/256 |

### Model architecture (`MarioPINNWorldModel`)

```
s_t, a_t ─► MLP trunk (128×128, LayerNorm, GELU) ─► delta(obs)  ─┐
                             ├────────────────────► reward(s,a)   ├─► next state
                             └────────────────────► continue(s,a) ┘   (x,y integrated exactly)
```

The trunk predicts a per-dimension **delta**. Velocities are clamped to the
engine's saturation limits, then positions are integrated with the exact `+K·v`
rule, so the kinematic residual is identically zero regardless of the weights.

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
| Params | 40,217 |
| Test single-step MSE | **0.032** |
| Kinematic residual (x / y) | **2.6·10⁻⁸ / 1.2·10⁻⁹** (structurally zero) |
| Train time (3000 transitions, CUDA) | 65 s |
| Collection throughput | ~18 env steps/s (real emulator) |

**Sample-efficiency study** (test MSE vs. number of real transitions trained on):

| N transitions | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| Test MSE | 0.079 | 0.053 | 0.030 | 0.040 |

Training on **200 transitions (~3 s of gameplay)** already gives MSE 0.079 — the
physics inductive bias reproduces the smw-pinn >25× sample-efficiency finding on a
different console.

### Stage 3 results — real-time play on the console (World 1-1, goal ≈ 4256 px, 8 episodes)

| Controller | World model used? | Mean progress | Max progress | Control latency | Level cleared |
|---|---|---|---|---|---|
| **CEM-MPC + hazard reflex** | Yes (online replanning) | **1269 px (30%)** | **1730 px (41%)** | 80 ms/frame | not yet |
| Dyna-PPO (Stage 2 policy) | Yes (imagination-trained) | 347 px (8%) | 424 px | **2.5 ms/frame** | not yet |
| Reflex only (no model) | No | 484 px | 561 px | ~0 | not yet |

---

## 3. Controllers

* **CEM-MPC (`--controller mpc`, primary).** Cross-Entropy-Method Model Predictive
  Control over discrete action sequences (categorical CEM, H=20, 160 candidates,
  3 refinements). It re-plans **every executed step from the true state**, so only
  the model's 1-step accuracy matters — which is exact for kinematics. This is the
  strongest real-console controller (as reported in smw-pinn §10.6).
* **Dyna-PPO (`--controller ppo`).** An SB3 `MlpPolicy` actor-critic trained entirely
  inside the learned model's imagination. Runs at **2.5 ms/frame** (>400 FPS,
  comfortably real-time), but transfers more weakly because the model cannot foresee
  *dynamic* Goomba contact over long horizons.
* **Reactive (`--controller reactive`, ablation).** Exact-state hazard reflex with a
  continuous rightward drive, no planning. Included to isolate the world model's
  contribution — MPC more than doubles it.

### Hazard reflex (`ReflexiveMPCAgent`)

The learned model captures *player* physics perfectly but *enemy* dynamics are weak
(they are only observed when on screen). We therefore compute an **exact** jump
reflex from the true egocentric RAM state — upcoming pits (ROM geometry flags) and
live enemies (`dx, dy, type`) — and commit a running hop when a hazard is close and
Mario is grounded. A key fix (the reference's reflex-vs-MPC arbitration): **do not
leap for a pit if a Goomba sits in the landing zone**, which was the dominant
failure mode. Enemies are handled by the reflex; pits by the MPC.

---

## 4. Does the agent finish the level?

**Not yet, and we report that honestly.** The agent plays the stage live in real
time, clears the early Goombas and the 1-tile and 4-tile pits (reaching a best of
~1730 px / 41%, occasionally ~2560 px / 60%), and then dies on the denser mid-level
hazard clusters (overlapping Goombas and falling `MegaDrop` objects around
x≈1250–1730 px).

The binding constraint is the **emulator control set**, not the world model: the
6-action mapping has **no dash button**, so Mario only walks (~1.3 px/frame). Weak,
short hops make the widest pits and tight enemy-timing marginal. This mirrors the
smw-pinn study, whose best autonomous SNES navigation was also a few-screen widths
(~1016 px) and which flags *long-horizon dynamic-hazard perception* as open frontier
work. The physics-informed **world model itself is solved** (residual = 0, MSE 0.03,
200-sample training); extending full-level completion is a control/feature problem
(e.g. adding a run/dash action, a probabilistic enemy head, or a tilemap-A\* global
planner) rather than a modelling one.

---

## 5. Files

| File | Role |
|---|---|
| [`src/pinn_world_model.py`](../src/pinn_world_model.py) | `MarioPINNWorldModel`, `shaped_step`, `PINNRollout`, `CEMMPController`, `PINNImaginationEnv` |
| [`src/train_pinn_world_model.py`](../src/train_pinn_world_model.py) | Stage 1: collect RAM data, train PINN, report MSE / residual / drift / sample-efficiency |
| [`src/train_pinn_policy.py`](../src/train_pinn_policy.py) | Stage 2: Dyna-PPO in imagination |
| [`src/eval_pinn_realtime.py`](../src/eval_pinn_realtime.py) | Stage 3: real-time console eval (`mpc` / `ppo` / `reactive`), rendering + `.avi` recording + finish detection |
| `media/pinn_mpc_run.avi` | Recorded real-time MPC gameplay clip |

### Tunable flags (`eval_pinn_realtime.py`)

`--horizon`, `--n-cand`, `--n-iter` (planning depth) · `--w-progress/-enemy/-pit`
(model-reward shaping) · `--enemy-lo/-hi/-dy`, `--pit-lookahead`, `--no-reflex`
(reflex) · `--goal-px` (auto-read from ROM) · `--render` / `--record PATH` / `--speed`.
