# Project Status & Gap Analysis

*Last updated: 2026-09-24. ROM: NSMB DS EUR (NTR-A2DP-EUR). All code lives under `src/`.*

---

## Executive Summary

The pipeline goes from raw DS emulator → RAM telemetry → physics-informed world model → real-time agent. All three PINN pipeline stages are executable and produce metrics. **No controller has cleared 1-1 yet** (max progress ≈ 1730 px out of 4256 px goal). The dominant failure modes are pit deaths (the level-geometry planner infers pits from tile data but the agent still misjudges enemy-blocked approaches) and a systematic early-death cluster at ~336 px in the PPO and reactive baselines caused by the first Goomba on flat ground.

---

## 1. What Is Done ✅

### 1.1 Environment & RAM Telemetry (`src/ram_env.py`, `src/ram_state.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| DeSmuME Python binding (`py-desmume`) | ✅ | Fallback stubs for offline testing |
| Mario RAM addresses (EUR) | ✅ | Lives `0x0209DC00`, cam `0x02098240/0x020DCFA0`, odometer `0x0209AE9C`, Mario base `0x021C1890`, velocity addrs validated |
| Object type/A(Y)/B(X) at `+0x44/0x60/0x68` | ✅ | Validated with differential search + screenshots + AR codes |
| Enemy linked-list scanner (`SCAN_LO`–`SCAN_HI`) | ✅ | Finds up to 3 closest enemies by `|dx|`; links at ~20 steps proximity |
| 17-dim RAM observation vector | ✅ | `[x, y, vx, vy, on_ground, lives, time, cam_x, 3×(edx,edy,etype)]` |
| 8-action space with dash | ✅ | Reverse-engineered: X/Y = dash; running jump ≈ 120 px clears wide pits |
| Reward: progress×2 − 0.05/step, death −15, clear +100 | ✅ | Terminal +100 reward on flagpole contact |
| Level-finish detection | ✅ | Hardware RAM flag isolated at `CLEAR_FLAG_ADDR = 0x020DCF27` (Area/Camera controller) + flagpole coordinate (`4032.0 px`) |
| Dynamic enemy velocities | ✅ | Finite-difference `(vx, vy)` tracked in `RamState.enemies()` |
| Absolute screen-pixel X | ⚠️ Partial | Two consistent bases (`0x021C1828` vs object B); **delta is correct** (reward works), but absolute X is ±40 px ambiguous |

### 1.2 ROM Level Parser (`src/course.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| `course/A##_#.bin` section table | ✅ | Header = u32 size + 14 (offset,length) pairs tiling the file |
| Sprite/object block (Block 6) | ✅ | 12 B/record: `type u16, x u16, y u16 (TILES), data 6B` ended by `0xFFFFFFFF` |
| Entrance block (Block 5) | ✅ | 20 B/record: x,y,cam in pixels |
| `_bgdat.bin` tile objects | ✅ | All walkable ground/slope/pipe/stair objects included; false pits eliminated |
| Calibration vs RAM (1-1) | ✅ | Spawn tile 3 → RAM 48 px, ground row 30 → RAM −480 px, Goomba tile 24 → RAM 365 px |
| All 382 course files parsed | ✅ | ndspy `NintendoDSRom`, 2449-file FS, manifest at `tools/fs_manifest.json` |
| JSON export for all levels | ✅ | 191 levels exported (`out/dataset/levels.jsonl`, `entities.csv` via `tools/build_dataset.py`) |
| Enemy-type ID mapping | ✅ (85%) | 342 profile slots extracted from CC0 decomp into `out/dataset/object_ids.json`, loaded by `src/course.py` |
| Level-finish tile detection | ✅ | Flagpole detected at tile 252 (`4032.0 px`) across course sprites |

### 1.3 Tilemap A\* Planner (`src/tilemap_planner.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| Ground/pit mask from `_bgdat.bin` | ✅ | `GROUND_OBJS` includes all solid surfaces; real 1-1 pits detected at width 2-3 tiles |
| Run-jump reachability | ✅ | Dynamic ballistic reach: $X_{reach} = v_x \cdot t_{airborne}$ ($t_{air} = 48\text{ frames}$) |
| A\* over tile columns | ✅ | Global layer; feeds takeoff column to CEM-MPC |
| Enemy-aware planning | ✅ | `enemy_threat_ahead()` checks takeoff/landing corridors; triggers `dash_leap` or `retreat_wait` |
| Multi-level support | ⚠️ Not tested | Only validated on 1-1 |

### 1.4 PINN World Model (`src/pinn_world_model.py`, `src/train_pinn_world_model.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| Hard-residual kinematics identity | ✅ | Verified: `x_{t+1} = x_t + vx_{t+1}/16`, residual mean ≈ 4×10⁻⁴ |
| MLP trunk (128×128, LayerNorm, GELU) | ✅ | Δobs head + reward head + continue head |
| Heteroscedastic `head_logvar` | ✅ | Gaussian-NLL loss; uncertainty used by pessimistic planner |
| Transition pool | ✅ | 3500 real transitions, `models/pinn_pool.npz` |
| Trained model | ✅ | `models/pinn_nsmb.pt`, 43 440 params, test MSE 0.0558 |
| Open-loop drift | ⚠️ Moderate | Mean drift 1738 px over 44-frame horizon across 18 episodes → model diverges in extended rollouts |
| Sample efficiency | ✅ | MSE at 500 transitions ≈ MSE at 2000 (plateau early) |

### 1.5 PINN Policy & Real-Time Eval (`src/train_pinn_policy.py`, `src/eval_pinn_realtime.py`)

| Component | Status | Notes |
|-----------|--------|-------|
| Dyna-PPO in imagination | ✅ | 200 k steps, 8 imagined envs, 222 steps/s, `models/pinn_policy.zip` |
| CEM-MPC controller | ✅ | Re-plans every real step; 80 ms latency |
| Reactive baseline | ✅ | Rule-based: run right + jump on pit/enemy |
| **Results (n=8 each)** | | |
| — PPO | ❌ | Mean 347 px, max 424 px; dies at first Goomba |
| — Reactive | ❌ | Mean 484 px, max 561 px; truncation at 1000 steps |
| — MPC | ❌ | Mean 1269 px, max 1730 px; best so far but never clears pit cluster at ~1730 px |
| — Reactive (PINN obs) | ❌ | Mean 1137 px, max 1203 px; consistent Goomba deaths |
| Level clear | ❌ **Not achieved** | Goal: 4256 px |

### 1.6 Model-Free Baselines (`src/train_ppo_recurrent.py`, `src/train_ram.py`, `src/train_gnn.py`, etc.)

All 14 paradigms in the README are implemented. Benchmark results in `evals/benchmark_n30_merged.json` (n=30 stochastic + 10 deterministic, 14 models). No model clears the level in 100 k steps; best performers at 100 k: DrQ-v2, CURL, SPR in the 300–430 px range (stochastic mean); GNN (graph-based RAM) and RAM-PPO are working pipelines.

### 1.7 Experiment Tracking

- MLflow: `mlflow.db`, `mlruns.db`, `mlruns/` — all training runs logged
- TensorBoard: `tensorboard_logs/RecurrentPPO_0..20`
- `evals/`: JSON snapshots for every eval run

---

## 2. What Is Missing ❌

Listed in priority order for advancing the ML goal (clearing 1-1 or reaching a publishable result).

### P0 — Correctness blockers

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **Level-finish detection** | Agents cannot get the +100 reward for clearing; reward signal is truncated | ✅ Solved: Hardware flag isolated at `CLEAR_FLAG_ADDR = 0x020DCF27` via differential RAM search |
| **Absolute X ambiguity ±40 px** | Progress reward is correct (delta), but absolute position reporting in evals is off | Fix: anchor `B_OFFSET` once; compare `0x021C1828` spawn value with known entrance coordinate |
| **PPO policy collapses at ~336 px** | First Goomba on flat ground always kills the PPO agent (87 identical steps across 7/8 episodes → deterministic death loop) | Grounded contact mortality added to imagination + $25\times$ death loss weight in PINN |

### P1 — World model quality

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **Open-loop drift 1738 px over 44 frames** | MPC re-plans every step (mitigates this), but PPO relies on long imagined rollouts where the model is inaccurate | Increase transitions to 10k (currently 3500); add recurrent state in the PINN trunk (GRU head); or use an ensemble of 3 PINNs and take pessimistic lower bound |
| **Enemy dynamics unmodeled beyond relative position** | Enemies move; their relative position changes in ways the PINN doesn't capture well (logvar head marks this as high-uncertainty) | ✅ Solved: Dynamic enemy velocities `(vx, vy)` tracked in `RamState.enemies()` |
| **No contact/death prediction** | The PINN `continue` head is a binary; it doesn't distinguish death-by-enemy from death-by-pit | ✅ Solved: Explicit physical collision hitbox check in `PINNImaginationEnv` |

### P2 — Planning gaps

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **Enemy-unaware A\* planner** | MPC consistently dies at ~1730 px where a Goomba blocks the approach to a pit edge | ✅ Solved: `enemy_threat_ahead` added to `TilemapAStar` |
| **Multi-level generalization** | Planner and RAM addresses only validated on 1-1 | Run `course.py` batch export on all 382 levels → JSON; verify calibration on 1-2 and 2-1 |
| **JUMP_REACH_PX hardcoded** | 120 px works for most pits in 1-1 but breaks on wider pits in later worlds | ✅ Solved: Dynamic reach $X_{\text{reach}}(v_x) = v_x \cdot t_{\text{air}}$ ($t_{\text{air}} = 48\text{ frames}$) |

### P3 — Documentation & reproducibility

| Gap | Impact | Suggested Fix |
|-----|--------|---------------|
| **No formal `docs/` entry for RAM reverse engineering** | The discovery methodology lives only in `ram_state.py` | ✅ Solved: Documented in [`docs/REVERSE_ENGINEERING_REPORT.md`](REVERSE_ENGINEERING_REPORT.md) |
| **Complete Gap Analysis & Roadmap** | Gaps across all pillars not centralized | ✅ Solved: Documented in [`docs/COMPREHENSIVE_GAP_ANALYSIS_AND_ROADMAP.md`](COMPREHENSIVE_GAP_ANALYSIS_AND_ROADMAP.md) |
| **`evals/benchmark_n30_merged.json` not described** | The eval protocol (n=30, seeds, det/stoch split) is in the JSON but not in any doc | ✅ Solved: Documented in [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) |
| **No trained model cards** | Each `.zip`/`.pt` in `models/` has no metadata doc | ✅ Solved: Documented in [`docs/MODEL_CARDS.md`](MODEL_CARDS.md) |
| **README §14 (PINN) not linked to WORLD_MODEL_PINN.md** | The detailed doc exists but README only has a CLI snippet | Add link in README |

---

## 3. Current Research Status & Next Steps (ordered)

### Completed Milestones ✅
1. **Level-Finish Detection:** Hardware stage clear flag isolated at `CLEAR_FLAG_ADDR = 0x020DCF27` (transitions $0 \to 1$ on flagpole descent / castle entry).
2. **Dataset Export:** 191 stages parsed and normalized into [`out/dataset/levels.jsonl`](../out/dataset/levels.jsonl) and [`out/dataset/entities.csv`](../out/dataset/entities.csv) with corrected 20.12 fixed-point coordinate scaling.
3. **PINN & Policy Retraining:** $25\times$ death loss weighting + grounded physical collision hitbox mortality integrated into imagination.
4. **Player Jump State Machine Decomposition (Route B):** Executed 60 Hz single-frame telemetry captures on DeSmuME via [`tools/decompose_mario_jump.py`](../tools/decompose_mario_jump.py). Exact piecewise gravities ($g_{\text{ascent}} = 0.0919\text{ px/f}^2$ vs $g_{\text{descent}} = 0.2118\text{ px/f}^2$), launch velocity ($V_y = 3.7188\text{ px/f}$), apex hang time (3 frames @ frame 26), and maximum reach ($109.73\text{ px}$ @ $V_x = 2.136\text{ px/f}$) saved to [`out/dataset/physics_jump_state_machine.json`](../out/dataset/physics_jump_state_machine.json).
5. **Pipe Obstacle Anticipation in PPO+Reflex:** Integrated `TilemapAStar` terrain perception into `ReflexivePPOAgent`. Launch window anticipates obstacles $16-55\text{ px}$ ahead, ensuring takeoff before $X = 358\text{ px}$ for the 3-block pipe at $X = 384\text{ px}$. Ballistic trajectory reaches apex ($Y = 71.06\text{ px}$) over the pipe, eliminating wall collisions and advancing PPO+Reflex from $375.6\text{ px}$ to **$1,000.76\text{ px}$ mean** and **$1,139.46\text{ px}$ max**.
6. **Overcoming the 1,498 px Obstacle in MPC:** Extended CEM-MPC predictive rollout horizon to $H=24$ and implemented corridor-specific adaptive deceleration (`compound == "decelerate"` via Action 1: Walk Right) at cols 92–97 ($1472-1552\text{ px}$). Filtered non-hostile entities (MegaDrop, coins) and eliminated phantom row 0 ceiling markers in `Course.ground_cover`. CEM-MPC surpassed the $1,498\text{ px}$ barrier, reaching **$1,510.02\text{ px}$ max** ($1,268.16\text{ px}$ mean).
7. **Academic Documentation:** Added [`docs/REVERSE_ENGINEERING_REPORT.md`](REVERSE_ENGINEERING_REPORT.md), [`docs/MODEL_CARDS.md`](MODEL_CARDS.md), and [`docs/BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md).

---

## 4. Empirical Evaluation Summary Table

| Controller | Mean Progress (px) | Max Progress (px) | Clear Rate | Notes |
|---|:---:|:---:|:---:|---|
| **CEM-MPC (PINN, n=3)** | **1,268.2** | **1,510.0** | **0.0%** | Overcomes 1498 px barrier via extended $H=24$ horizon & patrol deceleration |
| **PPO+Reflex (PINN, n=5)** | **1,000.8** | **1,139.5** | **0.0%** | 100% clearance of 384 px pipe via TilemapAStar takeoff before 358 px |
| **Reactive (Rule-based, n=4)** | 1,136.7 | 1,202.7 | 0.0% | Rule-based hazard leaps; dies at 1115-1203 px |
| **PPO Pure (Pixels, n=10 det)** | 120.8 | 120.8 | 0.0% | Deterministic Goomba death loop at entrance |
| **Goal (Flagpole)** | **4,032.0** | **4,256.0** | **100%** | Flagpole contact & castle entrance (World 1-1) |


---

## 5. File Index

```
src/
  env.py                 Visual (CNN) environment — pixel-based obs, optical flow reward
  ram_env.py             RAM environment — 17-dim vector obs, no CNN
  ram_state.py           RAM address constants + RamState reader (validated EUR)
  course.py              ROM level parser (ndspy) — sprites, bgdat, entrances
  tilemap_planner.py     A* global planner from tile geometry
  pinn_world_model.py    Hard-residual PINN + CEM-MPC controller
  train_pinn_world_model.py  Stage 1: collect transitions + train PINN
  train_pinn_policy.py   Stage 2: Dyna-PPO in imagination
  eval_pinn_realtime.py  Stage 3: real-time eval (mpc / ppo / reactive)
  train_ram.py           RAM-PPO baseline
  train_gnn.py           Graph NN baseline
  train_ppo_recurrent.py RecurrentPPO (all augmentation variants)
  evaluate_benchmark.py  n=30 benchmark runner
  run_pipeline.py        Full pipeline orchestrator
  ...

models/
  pinn_nsmb.pt           PINN world model (43 440 params, MSE 0.056)
  pinn_pool.npz          3500 real transitions
  pinn_policy.zip        Dyna-PPO policy (200 k imagination steps)
  ram_ppo_100k.zip       RAM-PPO 100 k
  ram_geo_100k.zip       RAM + geometry features 100 k
  gnn_100k.zip           GNN 100 k
  ppo_icm_100k.zip       PPO + ICM 100 k
  ppo_icm_ae_100k.zip    PPO + ICM + Autoencoder 100 k
  ...

evals/
  pinn_worldmodel_metrics.json   PINN train/test metrics
  pinn_policy_metrics.json       Dyna-PPO training summary
  pinn_realtime_mpc.json         n=8 MPC real-time eval
  pinn_realtime_ppo.json         n=8 PPO real-time eval
  pinn_realtime_reactive.json    n=8 reactive real-time eval
  pinn_realtime_eval.json        n=4 reactive (PINN obs)
  benchmark_n30_merged.json      n=30 stochastic + n=10 det, 14 models

tools/
  export_levels.py               Batch parser for all 191 levels
  build_dataset.py               Emits entities.csv and levels.jsonl
  extract_profiles.py            Scrapes 342 actor profiles from CC0 decomp

out/dataset/
  levels.jsonl                   191 levels with full entity tables (0.70 MB)
  entities.csv                   5311 entities with corrected pixel coords (0.28 MB)
  object_ids.json                342 actor profile mappings
  physics.json                   20.12 fixed-point engine constants and laws
  meta.json                      Dataset schema specification

docs/
  WORLD_MODEL_PINN.md                         Deep-dive: PINN methodology
  PROJECT_STATUS.md                           This file
  REVERSE_ENGINEERING_REPORT.md               Comprehensive NSMB DS reverse engineering report
  COMPREHENSIVE_GAP_ANALYSIS_AND_ROADMAP.md   Master gap analysis across RE and ML pillars
```
