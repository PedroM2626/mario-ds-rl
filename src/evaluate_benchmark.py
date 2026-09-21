"""
Unified benchmark:
  N DETERMINISTIC + M STOCHASTIC episodes per model (--eps/--eps-stoch),
  full episodes (headless, no rendering), PARALLEL evaluation (--workers, 1 env per
  worker with staggered initialization). Auto-detects RecurrentPPO vs PPO vs MLP-RAM vs GNN.

Usage:
  python src/evaluate_benchmark.py --models pure,spr --eps 10 --eps-stoch 30 --workers 4
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import MarioNdsEnv

ROM = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
STATE = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"
MAX_STEPS = 5000  # Safety cap; natural episodes terminate on death/finish/timeout

MODELS = {
    "pure":       "models/ppo_pure_100k.zip",
    "ae_frozen":  "models/ppo_autoencoder_frozen_100k.zip",
    "curl":       "models/ppo_curl_online_100k.zip",
    "hybrid":     "models/ppo_hybrid_ae_curl_100k.zip",
    "spr":        "models/ppo_spr_100k.zip",
    "drq":        "models/ppo_drq_100k.zip",
    "recurrent":  "models/recurrent_ppo_mario.zip",
    "impala":     "models/mario_impala.zip",
    "icm":        "models/ppo_icm_100k.zip",
    "icm_ae":     "models/ppo_icm_ae_100k.zip",
    "ram100k":    "models/ram_ppo_100k.zip",
    "ramgeo100k": "models/ram_geo_100k.zip",
    "gnn100k":    "models/gnn_100k.zip",
}


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor as _BFE


class _DrQPassthrough(_BFE):
    """Replicates the structure of DrQFeaturesExtractorWrapper (from train_ppo_recurrent,
    which was function-local and thus not reconstructible during standard load).
    In evaluation mode (training=False), the original is pure passthrough, yielding identical inference."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=512)
        from stable_baselines3.common.torch_layers import NatureCNN
        self.base_extractor = NatureCNN(observation_space)

    def forward(self, observations):
        return self.base_extractor(observations)


def _load_drq_manual(path):
    """Fallback for DrQ models: builds RecurrentPPO with known kwargs
    (saved policy_kwargs omit the wrapper swapped post-construction in training)
    and loads policy.pth directly into the policy."""
    import io
    import zipfile
    import gymnasium as gym
    import torch
    from sb3_contrib import RecurrentPPO

    class _Stub(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (1, 84, 84), np.uint8)
        action_space = gym.spaces.Discrete(6)

        def reset(self, **kw):
            return np.zeros((1, 84, 84), np.uint8), {}

        def step(self, a):
            return np.zeros((1, 84, 84), np.uint8), 0.0, True, False, {}

    with zipfile.ZipFile(path) as z:
        raw = z.read("policy.pth")
    model = RecurrentPPO(
        "CnnLstmPolicy", _Stub(),
        policy_kwargs={"features_extractor_class": _DrQPassthrough,
                       "features_extractor_kwargs": {},
                       # Without this, SB3 resolves net_arch=None to [64,64] when
                       # custom extractor is present; DrQ training used [] (default without
                       # custom extractor, which was swapped post-construction)
                       "net_arch": [],
                       "lstm_hidden_size": 256,
                       "enable_critic_lstm": True})
    # In training, ONLY the shared extractor was replaced by the DrQ wrapper;
    # pi_/vf_ remained pure NatureCNN (unused in forward with share=True,
    # but present in state_dict). Replicate structure for strict load.
    from stable_baselines3.common.torch_layers import NatureCNN
    model.policy.pi_features_extractor = NatureCNN(model.observation_space)
    model.policy.vf_features_extractor = NatureCNN(model.observation_space)
    model.policy.to(model.device)
    sd = torch.load(io.BytesIO(raw), map_location=model.device)
    model.policy.load_state_dict(sd, strict=True)
    return model


def load_model(path):
    """Returns (model, kind). kind = 'recurrent' | 'ppo' | 'ram' | 'gnn'."""
    err_rec = err_ppo = None
    try:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path), "recurrent"
    except Exception as e:
        err_rec = e
    if "base_extractor" in str(err_rec):
        # Model trained with DrQ wrapper: manual reconstruction + policy.pth
        try:
            from sb3_contrib import RecurrentPPO  # noqa: F401 (ensure import)
            return _load_drq_manual(path), "recurrent"
        except Exception as e:
            err_rec = e
    try:
        from stable_baselines3 import PPO
        m = PPO.load(path)
        # MLP over RAM vector vs CNN: distinguished by policy class;
        # GNN (graph 134) vs RAM MLP (17/23) by observation shape.
        if type(m.policy).__name__ == "ActorCriticPolicy":
            if tuple(m.observation_space.shape) == (134,):
                return m, "gnn"
            return m, "ram"
        return m, "ppo"
    except Exception as e:
        err_ppo = e
    raise RuntimeError(f"Failed to load {path} as RecurrentPPO ({err_rec}) "
                       f"and as PPO ({err_ppo})")


def eval_ram(model, n_eps):
    raise NotImplementedError("use the worker pool (--workers) or _eval_box with dedicated env")


def eval_gnn(model, n_eps):
    raise NotImplementedError("use the worker pool (--workers) or _eval_box with dedicated env")


def _eval_box(model, env, n_eps, deterministic):
    """Generic loop for Box observations (ram/gnn) on an existing environment."""
    rewards, steps_list = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        total, steps, done = 0.0, 0, False
        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, trunc, _ = env.step(
                int(action) if np.ndim(action) == 0 else int(action[0]))
            done = bool(done or trunc)
            total += float(r)
            steps += 1
        rewards.append(total)
        steps_list.append(steps)
    return rewards, steps_list


# ---------------------------------------------------------------- pool
_W = {}
_ENVS = {}


def _w_init(counter, lock):
    import time as _t
    with lock:
        wid = int(counter.value)
        counter.value = wid + 1
    _W["wid"] = wid
    _t.sleep(wid * 4.0)  # Staggered init: concurrent DeSmuMEs collide


def _close_others(cache, keep=None):
    """Ensure no multiple live DeSmuMEs: close environments of other types."""
    for k, e in list(cache.items()):
        if k != keep:
            try:
                e.close()
            except Exception:
                pass
            del cache[k]


def _get_env(ekind):
    if ekind in _ENVS:
        return _ENVS[ekind]
    _close_others(_ENVS)
    if ekind == "rec":
        env = MarioNdsEnv(rom_path=ROM, state_path=STATE)
        assert env.has_emulator, "Emulator failed to initialize!"
    elif ekind == "vec":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
        venv = MarioNdsEnv(rom_path=ROM, state_path=STATE)
        assert venv.has_emulator, "Emulator failed to initialize!"
        env = VecFrameStack(DummyVecEnv([lambda: venv]), n_stack=4)
    elif ekind in ("ram17", "ram23"):
        from ram_env import MarioRamEnv
        env = MarioRamEnv(geo=(ekind == "ram23"))
        assert env.has_emulator, "Emulator failed to initialize!"
    elif ekind == "gnn":
        from graph_env import MarioGraphEnv
        env = MarioGraphEnv()
        assert env.has_emulator, "Emulator failed to initialize!"
    else:
        raise ValueError(f"Unknown ekind: {ekind}")
    _ENVS[ekind] = env
    return env


def _run_task(task):
    # (name, path, ekind_hint, mkind, mode, n_eps, seed)
    import random
    import torch
    name, path, ekind_hint, mkind, mode, n_eps, seed = task
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    model, kind = load_model(path)
    assert kind == mkind, f"{name}: expected {mkind}, received {kind}"
    if ekind_hint == "auto":
        if kind == "gnn":
            ekind = "gnn"
        else:
            ekind = f"ram{tuple(model.observation_space.shape)[0]}"
    else:
        ekind = ekind_hint
    env = _get_env(ekind)
    det = (mode == "det")
    if kind == "recurrent":
        r, s = eval_recurrent(model, env, n_eps, det)
    elif kind == "ppo":
        r, s = eval_ppo_framestack(model, env, n_eps, det)
    else:
        r, s = _eval_box(model, env, n_eps, det)
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return {"name": name, "mode": mode,
            "rewards": [float(v) for v in r], "steps": [int(v) for v in s]}


def eval_recurrent(model, env, n_eps, deterministic):
    rewards, steps_list = [], []
    for _ in range(n_eps):
        obs, _ = env.reset()
        lstm_states, ep_start = None, np.array([True])
        total, steps, done = 0.0, 0, False
        while not done and steps < MAX_STEPS:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=ep_start,
                deterministic=deterministic)
            obs, r, done, trunc, _ = env.step(int(action) if np.ndim(action) == 0
                                              else int(action[0]))
            done = bool(done or trunc)
            ep_start = np.array([done])
            total += float(r)
            steps += 1
        rewards.append(total)
        steps_list.append(steps)
    return rewards, steps_list


def eval_ppo_framestack(model, vec_env, n_eps, deterministic):
    rewards, steps_list = [], []
    for _ in range(n_eps):
        obs = vec_env.reset()
        total, steps, done = 0.0, 0, [False]
        while not done[0] and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, _ = vec_env.step(action)
            total += float(r[0])
            steps += 1
        rewards.append(total)
        steps_list.append(steps)
    return rewards, steps_list


def _stat(r, s):
    return {"rewards": [float(v) for v in r], "steps": [int(v) for v in s],
            "mean": float(np.mean(r)), "std": float(np.std(r)),
            "max": float(np.max(r)), "mean_steps": float(np.mean(s))}


def _print_res(name, mode, o):
    print(f"[{name}/{mode}] mean={o['mean']:.2f} std={o['std']:.2f} "
          f"max={o['max']:.2f} mean_steps={o['mean_steps']:.0f} "
          f"(n={len(o['rewards'])})", flush=True)


def _save_partial(fp, meta, results, t0):
    os.makedirs("evals", exist_ok=True)
    with open(fp, "w") as f:
        json.dump({"protocol": meta, "elapsed_s": time.time() - t0,
                   "results": results}, f, indent=1)


def _run_sequential(names, args, kind_of, results, t0, fp, meta):
    """One environment per type at a time (safe in sandbox; sequential)."""
    import random
    import torch
    groups = {}
    for i, name in enumerate(names):
        path = MODELS[name]
        if not os.path.exists(path):
            print(f"[{name}] SKIP (missing file: {path})", flush=True)
            continue
        ek, mk = kind_of(name)
        groups.setdefault(ek, []).append((i, name, path, mk))
    for ek, items in groups.items():
        env_cache = {}
        if ek != "auto":
            env_cache[ek] = _get_env_main(ek)
        try:
            for i, name, path, mk in items:
                model, kind = load_model(path)
                assert kind == mk, f"{name} is not {mk}!"
                if ek == "auto":
                    dim = tuple(model.observation_space.shape)[0]
                    key = "gnn" if kind == "gnn" else f"ram{dim}"
                    if key not in env_cache:
                        _close_others(env_cache, keep=key)
                        env_cache[key] = _get_env_main(key)
                    env = env_cache[key]
                else:
                    env = env_cache[ek]
                out = results.setdefault(name, {})
                for mode, n_eps, det, sd in (
                        ("det", args.eps, True, args.seed + 2 * i),
                        ("stoch", args.eps_stoch, False, args.seed + 2 * i + 1)):
                    random.seed(sd)
                    np.random.seed(sd % (2 ** 32))
                    torch.manual_seed(sd)
                    if mk == "recurrent":
                        r, s = eval_recurrent(model, env, n_eps, det)
                    elif mk == "ppo":
                        r, s = eval_ppo_framestack(model, env, n_eps, det)
                    else:
                        r, s = _eval_box(model, env, n_eps, det)
                    out[mode] = _stat(r, s)
                    _print_res(name, mode, out[mode])
                del model
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                _save_partial(fp, meta, results, t0)  # anti-crash checkpoint
        finally:
            for e in env_cache.values():
                try:
                    e.close()
                except Exception:
                    pass


def _build_once(ekind):
    if ekind == "rec":
        env = MarioNdsEnv(rom_path=ROM, state_path=STATE)
        assert env.has_emulator, "Emulator failed to initialize!"
        return env
    if ekind == "vec":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
        venv = MarioNdsEnv(rom_path=ROM, state_path=STATE)
        assert venv.has_emulator, "Emulator failed to initialize!"
        return VecFrameStack(DummyVecEnv([lambda: venv]), n_stack=4)
    if ekind in ("ram17", "ram23"):
        from ram_env import MarioRamEnv
        env = MarioRamEnv(geo=(ekind == "ram23"))
        assert env.has_emulator, "Emulator failed to initialize!"
        return env
    if ekind == "gnn":
        from graph_env import MarioGraphEnv
        env = MarioGraphEnv()
        assert env.has_emulator, "Emulator failed to initialize!"
        return env
    raise ValueError(ekind)


def _get_env_main(ekind, tries=3):
    import gc as _gc
    import time as _t
    last = None
    for a in range(tries):
        try:
            return _build_once(ekind)
        except Exception as e:
            last = e
            print(f"[env {ekind}] attempt {a+1}/{tries} failed ({e}); retrying in 5s...",
                  flush=True)
            _gc.collect()
            _t.sleep(5)
    raise RuntimeError(f"env {ekind} failed {tries}x: {last}")


def _run_pool(names, args, kind_of, results, t0, fp, meta):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import Lock, Value
    tasks = []
    for i, name in enumerate(names):
        path = MODELS[name]
        if not os.path.exists(path):
            print(f"[{name}] SKIP (missing file: {path})", flush=True)
            continue
        ek, mk = kind_of(name)
        tasks.append((name, path, ek, mk, "det", args.eps, args.seed + 2 * i))
        tasks.append((name, path, ek, mk, "stoch", args.eps_stoch, args.seed + 2 * i + 1))
    counter, lock = Value("i", 0), Lock()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_w_init,
                             initargs=(counter, lock)) as pool:
        for res in pool.map(_run_task, tasks):
            o = results.setdefault(res["name"], {})
            o[res["mode"]] = _stat(res["rewards"], res["steps"])
            _print_res(res["name"], res["mode"], o[res["mode"]])
            _save_partial(fp, meta, results, t0)  # anti-crash checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=",".join(MODELS),
                    help="comma-separated subset of models to evaluate")
    ap.add_argument("--eps", type=int, default=10, help="number of deterministic episodes")
    ap.add_argument("--eps-stoch", type=int, default=10, help="number of stochastic episodes")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel environments (0 = sequential in current process)")
    ap.add_argument("--seed", type=int, default=0, help="base seed (task i uses base+i)")
    args = ap.parse_args()
    names = [m.strip() for m in args.models.split(",") if m.strip()]

    def kind_of(name):
        p = MODELS[name]
        if name == "impala" or p.endswith("mario_impala.zip"):
            return "vec", "ppo"
        if "gnn_" in p:
            return "auto", "gnn"
        if "ram_" in p:
            return "auto", "ram"
        return "rec", "recurrent"

    tasks = []
    for i, name in enumerate(names):
        path = MODELS[name]
        if not os.path.exists(path):
            print(f"[{name}] SKIP (missing file: {path})", flush=True)
            continue
        ek, mk = kind_of(name)
        tasks.append((name, path, ek, mk, "det", args.eps, args.seed + 2 * i))
        tasks.append((name, path, ek, mk, "stoch", args.eps_stoch, args.seed + 2 * i + 1))

    results = {}
    t0 = time.time()
    os.makedirs("evals", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fp = f"evals/benchmark_{ts}.json"
    meta = {"det": args.eps, "stoch": args.eps_stoch,
            "workers": args.workers, "seed": args.seed}
    if args.workers <= 0:
        _run_sequential(names, args, kind_of, results, t0, fp, meta)
    else:
        _run_pool(names, args, kind_of, results, t0, fp, meta)

    print(f"RESULTS saved to {fp} ({time.time()-t0:.0f}s)", flush=True)
    print("TABLE det | stoch (mean +- std, max):", flush=True)
    for name, o in results.items():
        print(f"  {name}: det={o['det']['mean']:.2f}+-{o['det']['std']:.2f}"
              f" (max {o['det']['max']:.2f}) | stoch={o['stoch']['mean']:.2f}+-"
              f"{o['stoch']['std']:.2f} (max {o['stoch']['max']:.2f})", flush=True)


if __name__ == "__main__":
    main()
