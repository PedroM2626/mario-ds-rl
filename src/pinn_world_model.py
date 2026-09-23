"""Physics-Informed World Model (Hard-Residual PINN) for New Super Mario Bros. DS.

Ported in spirit from https://github.com/PedroM2626/smw-pinn (Super Mario World),
adapted to NSMB-DS RAM telemetry (`src/ram_env.py`, `src/ram_state.py`).

Core idea (Ha & Schmidhuber 2018 + PIML): learn the *residual* unmodelled forces
of the game and embed the engine's exact discrete Euler integration directly into
the computational graph, so the predicted player kinematics satisfy the console's
integration rule by construction (zero structural residual). This yields a world
model that is trained with only a few hundred real transitions and is accurate
enough for closed-loop planning (CEM-MPC) and amortised policy training (Dyna-PPO).

Empirically validated for NSMB-DS (see docs/WORLD_MODEL_PINN.md): under the env
`frameskip=8`, the RAM-normalised state obeys, for both axes,

    x_{t+1} = x_t + v_{x,t+1} / 16      (residual std ~ 0.009 in px/512 units)

which is exactly the identity hard-wired into `MarioPINNWorldModel` below.

Observation layout (Box(23)) produced by `MarioRamEnv(geo=True)`:
    0  x          (px / 512)
    1  y          (px / 512)
    2  vx         (px/frame / 4)
    3  vy         (px/frame / 4)
    4  on_ground  (0/1)
    5  lives      (/10)
    6  time_left  (/400)
    7  cam        cumulative progress (px / 512)
    8..16          3 nearest enemies  (dx/256, dy/256, type/256)
    17..22         6 binary pit flags (columns +1..+6)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

# --- fixed state-space geometry -----------------------------------------
OBS_DIM = 23
N_ACTIONS = 6
IDX_X, IDX_Y, IDX_VX, IDX_VY = 0, 1, 2, 3
KINEMATIC = (IDX_X, IDX_Y, IDX_VX, IDX_VY)
# Normalised Euler step factor: x_{t+1} = x_t + K * v_{x,t+1}, K = 1/16.
EULER_K = 1.0 / 16.0
# Velocity saturation limits in normalised units (px/frame / 4). Generous bounds
# measured from random-policy data; clamped velocities cannot exceed the engine.
VX_LIMIT = 2.0
VY_UP_LIMIT = 3.0
VY_DOWN_LIMIT = 4.0


def encode_action(action: int) -> np.ndarray:
    a = np.zeros(N_ACTIONS, dtype=np.float32)
    a[int(action)] = 1.0
    return a


def encode_action_t(action: torch.Tensor) -> torch.Tensor:
    """Discrete integer/batch -> one-hot float tensor on same device/dtype."""
    return F.one_hot(action.long(), num_classes=N_ACTIONS).float()


def _mlp(sizes, hidden_act=nn.GELU, use_ln=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if use_ln:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(hidden_act())
    return nn.Sequential(*layers)


class ShapingConfig:
    """Weights for the analytic hazard/progress terms added on top of the learned
    reward when planning/imagining. Enemies and pits are read directly from the
    predicted egocentric state, making the controller robust to reward-model error.

    State index convention (MarioRamEnv geo=True): cam=7, on_ground=4,
    enemies (dx,dy,type) at (8,9,10),(11,12,13),(14,15,16); pit flags 17..22.
    """

    def __init__(self, progress=4.0, enemy=3.5, pit=1.5, step=0.02,
                 enemy_x_sigma=48.0, enemy_y_sigma=42.0):
        self.progress = progress
        self.enemy = enemy
        self.pit = pit
        self.step = step
        self.ex_s = enemy_x_sigma
        self.ey_s = enemy_y_sigma


ENEMY_DX = [8, 11, 14]
ENEMY_DY = [9, 12, 15]
ENEMY_T = [10, 13, 16]


def shaped_step(model, s, action, cfg: ShapingConfig):
    """One model step returning (next_s, shaped_reward, continue_prob).

    shaped_reward = learned_reward
                     + progress * cam_delta
                     - enemy    * sum_k active_k * gauss(dx) * gauss(dy)
                     - pit      * on_ground * (1 - flag[nearest ahead])
                     - step     * 1
    All terms are vectorised over the leading batch dimension.
    """
    a = encode_action_t(action).to(s.dtype)
    ns, r, logit_cont = model(s, a)
    p = torch.sigmoid(logit_cont)
    cam_delta = (ns[:, 7] - s[:, 7]).clamp(min=0.0)
    ex = ns[:, ENEMY_DX] * 256.0            # px, egocentric
    ey = ns[:, ENEMY_DY] * 256.0
    et = ns[:, ENEMY_T] * 256.0             # raw actor type (0 => empty slot)
    active = (et.abs() > 1.0).float()
    gx = torch.exp(-((ex.abs() / cfg.ex_s) ** 2))
    gy = torch.exp(-((ey.abs() / cfg.ey_s) ** 2))
    enemy_danger = (active * gx * gy).sum(dim=-1)
    pit_danger = ns[:, 4] * (1.0 - ns[:, 17])  # on ground with a pit in the next column
    shaped = (r + cfg.progress * cam_delta
              - cfg.enemy * enemy_danger
              - cfg.pit * pit_danger
              - cfg.step)
    return ns, shaped, p


class MarioPINNWorldModel(nn.Module):
    """Hard-residual physics-informed forward model with reward + continue heads.

    The trunk consumes the current state and the one-hot action and emits a
    per-dimension *delta*. Non-kinematic dimensions are updated residually, while
    the velocity dimensions are clamped to the engine's saturation limits and the
    positions are integrated analytically with the exact `+ K*v` rule. This makes
    the discrete-kinematics residual identically zero, regardless of weights.
    """

    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hid=128, layers=2,
                 k=EULER_K):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.k = k
        sizes = [obs_dim + n_actions] + [hid] * layers
        self.trunk = _mlp(sizes)
        self.head_delta = nn.Linear(hid, obs_dim)
        self.head_reward = nn.Sequential(nn.Linear(hid, hid // 2), nn.GELU(),
                                         nn.Linear(hid // 2, 1))
        self.head_continue = nn.Sequential(nn.Linear(hid, hid // 2), nn.GELU(),
                                           nn.Linear(hid // 2, 1))

    def forward(self, s, a_onehot):
        z = torch.cat([s, a_onehot], dim=-1)
        h = self.trunk(z)
        delta = self.head_delta(h)
        reward = self.head_reward(h).squeeze(-1)
        logit_cont = self.head_continue(h).squeeze(-1)

        ns = s + delta  # residual update for all dims
        vx = torch.clamp(s[:, IDX_VX] + delta[:, IDX_VX], -VX_LIMIT, VX_LIMIT)
        vy = torch.clamp(s[:, IDX_VY] + delta[:, IDX_VY], -VY_UP_LIMIT, VY_DOWN_LIMIT)
        ns = ns.clone()
        ns[:, IDX_VX] = vx
        ns[:, IDX_VY] = vy
        ns[:, IDX_X] = s[:, IDX_X] + vx * self.k   # EXACT integration, zero residual
        ns[:, IDX_Y] = s[:, IDX_Y] + vy * self.k
        return ns, reward, logit_cont

    @torch.no_grad()
    def step_batch(self, s, actions):
        """s:(B,obs) actions:(B,) integers -> (next_s, reward, continue_prob)."""
        a = encode_action_t(actions).to(s.dtype).to(s.device)
        ns, r, lc = self.forward(s, a)
        return ns, r, torch.sigmoid(lc)


class PINNRollout:
    """Batched discounted-reward rollouts through the learned world model.

    Used by both the CEM-MPC planner and the Dyna-PPO imagination env.
    """

    def __init__(self, model: MarioPINNWorldModel, gamma=0.99, device="cpu", cfg=None):
        self.model = model
        self.gamma = gamma
        self.device = device
        self.cfg = cfg or ShapingConfig()

    @torch.no_grad()
    def discounted_return(self, s0, action_seq):
        """s0:(B,obs); action_seq:(B,H) ints -> shaped returns (B,), cont prob (B,)."""
        B, H = action_seq.shape
        s = s0.to(self.device).float()
        returns = torch.zeros(B, device=self.device)
        discount = 1.0
        cont = torch.ones(B, device=self.device)
        for t in range(H):
            a = action_seq[:, t]
            s, r, p = shaped_step(self.model, s, a, self.cfg)
            returns = returns + discount * cont * r
            cont = cont * p
            discount *= self.gamma
        return returns, cont


class CEMMPController:
    """Cross-Entropy-Method MPC over discrete action sequences (categorical CEM).

    Re-plans every executed step from the *true* observed state (closed loop),
    which is robust to long-horizon model drift -- the strongest real-console
    controller reported in the smw-pinn study.
    """

    def __init__(self, model, horizon=16, n_candidates=96, n_elite=12,
                 n_iters=3, gamma=0.99, device="cpu", seed=None, cfg=None):
        self.model = model
        self.H = horizon
        self.N = n_candidates
        self.Kel = n_elite
        self.iters = n_iters
        self.cfg = cfg or ShapingConfig()
        self.roll = PINNRollout(model, gamma=gamma, device=device, cfg=self.cfg)
        self.device = device
        self.g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    @torch.no_grad()
    def plan(self, obs, best_prev=None):
        """obs:(obs_dim,) or (1,obs_dim) -> best action index."""
        s0 = torch.as_tensor(np.asarray(obs).reshape(1, -1), device=self.device).float()
        probs = torch.full((self.H, N_ACTIONS), 1.0 / N_ACTIONS, device=self.device)
        if best_prev is not None:
            # warm start: bias toward the previous plan
            probs = 0.6 * probs + 0.4 * best_prev
        for _ in range(self.iters):
            dist = torch.distributions.Categorical(probs=probs.expand(self.N, self.H, N_ACTIONS).reshape(-1, N_ACTIONS))
            seq = dist.sample().reshape(self.N, self.H)
            returns, _ = self.roll.discounted_return(s0.expand(self.N, -1), seq)
            elite = torch.topk(returns, self.Kel).indices
            eseq = seq[elite]
            counts = F.one_hot(eseq, num_classes=N_ACTIONS).float().mean(dim=0)
            probs = 0.7 * counts + 0.3 * probs  # smoothing
            probs = probs.clamp(0.02, 1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        best = probs.argmax(dim=-1)[0].item()
        return int(best), probs


class PINNImaginationEnv(gym.Env):
    """Gymnasium env that runs purely inside the learned world model.

    Provides the Dyna-PPO / MBPO training substrate: it resets to a real state
    sampled from a replay pool and advances with the model's predictions. A short
    fixed rollout length keeps imagination grounded (MBPO) to limit compounding
    model error.
    """

    metadata = {"render_modes": []}

    def __init__(self, model, pool_states, horizon=256, gamma=0.99, device="cpu",
                 reset_mix=0.5, seed=0, cfg=None):
        super().__init__()
        self.model = model
        self.cfg = cfg or ShapingConfig()
        self.pool = np.asarray(pool_states, dtype=np.float32)
        self.horizon = horizon
        self.device = device
        self.reset_mix = reset_mix
        self.observation_space = spaces.Box(-10.0, 10.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.rng = np.random.default_rng(seed)
        self.s = None
        self.t = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # reset_mix fraction starts at the true level-start state, rest from pool
        if self.rng.random() < self.reset_mix or len(self.pool) == 0:
            self.s = self.pool[0].copy() if len(self.pool) else np.zeros(OBS_DIM, np.float32)
        else:
            self.s = self.pool[self.rng.integers(len(self.pool))].copy()
        self.t = 0
        return self.s.copy(), {}

    @torch.no_grad()
    def step(self, action):
        s0 = torch.as_tensor(self.s[None], device=self.device).float()
        a = torch.as_tensor([int(action)], device=self.device)
        ns, r, p = shaped_step(self.model, s0, a, self.cfg)
        ns = ns.squeeze(0).clamp(-10.0, 10.0).cpu().numpy().astype(np.float32)
        self.t += 1
        cont = float(p.item())
        terminated = (cont < 0.5)
        truncated = self.t >= self.horizon
        # keep x,y,cam monotone-safe for the planner
        ns[IDX_X] = max(ns[IDX_X], -1.0)
        self.s = ns
        reward = float(r.item())
        return ns, reward, terminated, truncated, {"p_cont": cont}

    def render(self):  # pragma: no cover - imagination has no pixels
        return None

    def close(self):  # pragma: no cover
        pass
