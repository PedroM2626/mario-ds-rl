"""Stage 2: Dyna-PPO / MBPO -- train the actor-critic entirely inside the learned
PINN world model (imagination), then deploy zero-shot on the real emulator.

Because imagination needs no emulator, hundreds of thousands of transitions are
generated in seconds; the compact MLP policy transfers to the console in <1 ms.

Usage
-----
    python src/train_pinn_policy.py --timesteps 300000
Outputs
-------
    models/pinn_policy.zip        (Stable-Baselines3 PPO policy)
    evals/pinn_policy_metrics.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_world_model import (  # noqa: E402
    MarioPINNWorldModel, PINNImaginationEnv, ShapingConfig,
)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


def load_world_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = MarioPINNWorldModel(hid=ck["hid"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/pinn_nsmb.pt")
    ap.add_argument("--pool", default="models/pinn_pool.npz")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=200, help="imagination episode length (MBPO)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="models/pinn_policy.zip")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--w-progress", type=float, default=3.0)
    ap.add_argument("--w-enemy", type=float, default=5.0, help="Weight on enemy avoidance penalty")
    ap.add_argument("--w-pit", type=float, default=2.5, help="Weight on pit avoidance penalty")
    args = ap.parse_args()

    device = args.device
    model = load_world_model(args.model, device)
    cfg = ShapingConfig(progress=args.w_progress, enemy=args.w_enemy, pit=args.w_pit, enemy_x_sigma=64.0)
    pool = np.load(args.pool)
    # seed states: use real observed states (skip degenerate zeros)
    states = pool["S"]
    print(f"[pinn-pol] world model loaded; {len(states)} imagination seed states")

    env_fns = [lambda i=i: PINNImaginationEnv(
        model, states, horizon=args.horizon, device=device, cfg=cfg,
        reset_mix=0.3, seed=args.seed + i) for i in range(args.n_envs)]
    venv = DummyVecEnv(env_fns)

    policy = PPO(
        "MlpPolicy", venv, n_steps=512, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, learning_rate=3e-4,
        ent_coef=0.01, device=device, seed=args.seed, verbose=1,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    t0 = time.time()
    policy.learn(total_timesteps=args.timesteps)
    dt = time.time() - t0
    policy.save(args.out)

    metrics = {"timesteps": args.timesteps, "train_time_s": dt,
               "steps_per_s": args.timesteps / dt, "n_envs": args.n_envs,
               "imagination": "PINN world model (no emulator)"}
    os.makedirs("evals", exist_ok=True)
    with open("evals/pinn_policy_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[pinn-pol] trained in {dt:.1f}s ({args.timesteps/dt:,.0f} imag steps/s) -> {args.out}")


if __name__ == "__main__":
    main()
