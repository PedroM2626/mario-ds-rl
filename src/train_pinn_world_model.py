"""Stage 1: collect NSMB-DS RAM transitions and train the hard-residual PINN
world model. The model is intentionally trained from a *tiny* number of real
transitions (physics inductive bias => extreme sample efficiency).

Usage
-----
    python src/train_pinn_world_model.py --transitions 4000 --epochs 400
    python src/train_pinn_world_model.py --transitions 200            # sample-efficiency probe

Outputs
-------
    models/pinn_nsmb.pt           trained world-model weights + normalisation/config
    models/pinn_pool.npz          real transition pool (imagination seeds) + metrics
    evals/pinn_worldmodel_metrics.json
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ram_env import MarioRamEnv  # noqa: E402
from pinn_world_model import (  # noqa: E402
    MarioPINNWorldModel, OBS_DIM, N_ACTIONS, IDX_X, IDX_Y, IDX_VX, IDX_VY, EULER_K,
)

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
STATE = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"


@contextlib.contextmanager
def _silence_c_stdout():
    """DeSmuME logs boot/savestate chatter on the C-level stdout; hide it."""
    try:
        out_fd = sys.stdout.fileno()
        saved = os.dup(out_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        sys.stdout.flush()
        os.dup2(devnull, out_fd)
        os.close(devnull)
        yield
        sys.stdout.flush()
        os.dup2(saved, out_fd)
        os.close(saved)
    except Exception:
        yield


def collect(env, n_transitions, behavior="forward", seed=0):
    """Roll the real emulator with a light behaviour policy and record
    (s, a, s_next, reward, terminated) transitions."""
    rng = np.random.default_rng(seed)
    if behavior == "forward":
        # emphasise right + dash + running jumps so the model sees high-speed data
        # order: 0 noop,1 right,2 right+jump,3 right+dash,4 right+dash+jump,5 left,6 dash+jump,7 jump
        p = np.array([0.05, 0.20, 0.13, 0.20, 0.30, 0.03, 0.03, 0.03])
    else:
        p = np.full(N_ACTIONS, 1.0 / N_ACTIONS)
    p = p / p.sum()
    S, A, S2, R, D, EP = [], [], [], [], [], []
    ep_id = 0
    with _silence_c_stdout():
        obs, _ = env.reset()
        while len(S) < n_transitions:
            a = int(rng.choice(N_ACTIONS, p=p))
            obs2, r, term, trunc, _ = env.step(a)
            S.append(obs.astype(np.float32))
            A.append(a)
            S2.append(obs2.astype(np.float32))
            R.append(np.float32(r))
            D.append(np.float32(1.0 if term else 0.0))
            EP.append(ep_id)
            obs = obs2
            if term or trunc:
                obs, _ = env.reset()
                ep_id += 1
    return (np.stack(S), np.array(A, np.int64), np.stack(S2),
            np.array(R, np.float32), np.array(D, np.float32), np.array(EP, np.int64))


def make_tensors(data, device):
    S, A, S2, R, D = data[:5]
    a_oh = F.one_hot(torch.as_tensor(A), N_ACTIONS).float()
    return (torch.as_tensor(S, device=device), a_oh.to(device),
            torch.as_tensor(S2, device=device), torch.as_tensor(R, device=device),
            torch.as_tensor(D, device=device))


def split_temporal(data, val_frac=0.1, test_frac=0.15):
    n = len(data[0])
    i_tr = int(n * (1 - val_frac - test_frac))
    i_va = int(n * (1 - test_frac))
    def sl(lo, hi):
        return tuple(arr[lo:hi] for arr in data)
    return sl(0, i_tr), sl(i_tr, i_va), sl(i_va, n)


def train_model(data, model, device, epochs, lr=1e-3, wd=1e-5, bs=64, verbose=True):
    tr, va, _ = data
    St, At, S2t, Rt, Dt = make_tensors(tr, device)
    Sv, Av, S2v, Rv, Dv = make_tensors(va, device)
    n = St.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    # emphasise planning-relevant dims
    dim_w = torch.ones(OBS_DIM, device=device)
    for i in (IDX_X, IDX_Y, IDX_VX, IDX_VY):
        dim_w[i] = 3.0
    dim_w[7] = 2.0
    hist = []
    best_val, best_state, best_ep = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            ns, r, lc, logvar = model(St[idx], At[idx])
            err = (ns - S2t[idx]) ** 2
            # heteroscedastic Gaussian NLL on the transition (learned uncertainty),
            # dimension-weighted toward the planning-critical kinematics.
            nll = 0.5 * torch.exp(-logvar) * err + 0.5 * logvar
            loss_dyn = (nll * dim_w).mean()
            loss_rew = F.mse_loss(r, Rt[idx])
            w_cont = torch.where(Dt[idx] > 0.5, torch.full_like(Dt[idx], 25.0), torch.ones_like(Dt[idx]))
            loss_cont = F.binary_cross_entropy_with_logits(lc, 1.0 - Dt[idx], weight=w_cont)
            loss = loss_dyn + loss_rew + 0.5 * loss_cont
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            ns, r, lc, _ = model(Sv, Av)
            val = F.smooth_l1_loss(ns, S2v).item() + F.mse_loss(r, Rv).item()
        sched.step(val)
        hist.append(val)
        if val < best_val:
            best_val, best_ep = val, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d} | train {tot/n:.5f} | val {val:.5f} | best {best_val:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val


@torch.no_grad()
def eval_metrics(model, data, device, rollout_len=120):
    """Single-step MSE, exact-kinematic residual, and open-loop rollout drift.

    Drift is measured strictly within contiguous episodes (never across a reset)
    so it reflects genuine model error, not episode-boundary discontinuities.
    """
    _, _, test = data
    S, A, S2, R, D = test[:5]
    EP = test[5] if len(test) > 5 else np.zeros(len(S), np.int64)
    model.eval()
    St = torch.as_tensor(S, device=device)
    a_oh = F.one_hot(torch.as_tensor(A, device=device), N_ACTIONS).float()
    ns, r, lc, logvar = model(St, a_oh)
    mse = F.mse_loss(ns, torch.as_tensor(S2, device=device)).item()
    vx = ns[:, IDX_VX]; vy = ns[:, IDX_VY]
    rx = (ns[:, IDX_X] - St[:, IDX_X] - vx * EULER_K).abs().mean().item()
    ry = (ns[:, IDX_Y] - St[:, IDX_Y] - vy * EULER_K).abs().mean().item()
    # open-loop drift within episodes only
    ep_starts = {}
    for i, e in enumerate(EP):
        ep_starts.setdefault(int(e), []).append(i)
    rng = np.random.default_rng(0)
    drift, horizons = [], []
    for e, idxs in list(ep_starts.items()):
        if len(idxs) < 10:
            continue
        for _ in range(3):
            t0 = idxs[rng.integers(0, max(1, len(idxs) - 10))]
            span = min(rollout_len, idxs[-1] - t0)
            if span < 5:
                continue
            s = St[t0:t0 + 1].clone()
            for t in range(span):
                ai = torch.as_tensor([A[t0 + t]], device=device)
                s = model.step_batch(s, ai)[0]
            gt_x = S[t0 + span, IDX_X]; gt_y = S[t0 + span, IDX_Y]
            drift.append(np.hypot(s[0, IDX_X].item() - gt_x, s[0, IDX_Y].item() - gt_y) * 512)
            horizons.append(span)
    return {"test_mse": mse, "kinematic_residual_x": rx, "kinematic_residual_y": ry,
            "open_loop_drift_px_mean": float(np.mean(drift)) if drift else float("nan"),
            "open_loop_horizon_frames": float(np.mean(horizons)) if horizons else 0.0,
            "n_drift_episodes": len(drift)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default=ROM); ap.add_argument("--state", default=STATE)
    ap.add_argument("--transitions", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--sample-efficiency", action="store_true",
                    help="also train on N=200,500,1000,2000 and report test MSE")
    ap.add_argument("--pool-in", default=None, help="load existing pool npz instead of collecting")
    ap.add_argument("--out", default="models/pinn_nsmb.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    print(f"[pinn-wm] device={device}")

    t0 = time.time()
    if args.pool_in and os.path.exists(args.pool_in):
        p = np.load(args.pool_in)
        data = (p["S"], p["A"], p["S2"], p["R"], p["D"], p["EP"])
        collect_time = 0.0
        print(f"[pinn-wm] loaded {len(data[0])} transitions from {args.pool_in}")
    else:
        env = MarioRamEnv(rom_path=args.rom, state_path=args.state, geo=True)
        data = collect(env, args.transitions, seed=args.seed)
        collect_time = time.time() - t0
        env.close()
        print(f"[pinn-wm] collected {len(data[0])} real transitions in {collect_time:.1f}s "
              f"({len(data[0])/collect_time:.0f} steps/s)")

    splits = split_temporal(data)
    print(f"[pinn-wm] train/val/test = {len(splits[0][0])}/{len(splits[1][0])}/{len(splits[2][0])}")

    model = MarioPINNWorldModel(hid=args.hid).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pinn-wm] trainable params: {n_params}")
    print("[pinn-wm] training (physics-informed hard-residual PINN)...")
    t1 = time.time()
    model, hist, best_val = train_model(splits, model, device, args.epochs)
    train_time = time.time() - t1
    print(f"[pinn-wm] trained in {train_time:.1f}s | best val {best_val:.5f}")

    metrics = eval_metrics(model, splits, device)
    metrics.update({"collect_time_s": collect_time, "train_time_s": train_time,
                    "n_transitions": len(data[0]), "n_params": int(n_params)})
    print("[pinn-wm] metrics:", json.dumps(metrics, indent=2))

    if args.sample_efficiency:
        se = {}
        for nsub in (200, 500, 1000, 2000):
            if nsub >= len(data[0]):
                continue
            subdata = tuple(arr[:nsub] for arr in data)
            sub_split = split_temporal(subdata)
            m2 = MarioPINNWorldModel(hid=args.hid).to(device)
            m2, _, _ = train_model(sub_split, m2, device, epochs=200, verbose=False)
            se[nsub] = eval_metrics(m2, sub_split, device)["test_mse"]
            print(f"  sample-eff N={nsub:5d} -> test MSE {se[nsub]:.5f}")
        metrics["sample_efficiency_mse"] = se

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "obs_dim": OBS_DIM,
                "n_actions": N_ACTIONS, "hid": args.hid, "metrics": metrics}, args.out)
    np.savez_compressed("models/pinn_pool.npz", S=data[0], A=data[1], S2=data[2],
                        R=data[3], D=data[4], EP=data[5])
    os.makedirs("evals", exist_ok=True)
    with open("evals/pinn_worldmodel_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[pinn-wm] saved -> {args.out}, models/pinn_pool.npz")


if __name__ == "__main__":
    main()
