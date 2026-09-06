"""
World Models GA v2 para Mario DS:
  V: Encoder HIBRIDO (AE pre-treino + CURL online) extraido de
     models/ppo_hybrid_ae_curl_100k.zip  (ou --encoder autoencoder p/ baseline VAE)
  M: LSTM prevendo z_{t+1} | (z_t, a_t)
  C: sep-CMA-ES (CMA-ES diagonal, Ros & Hansen 2008) em vez de GA simples

Aceleracoes (item 2):
  - avaliacoes da populacao em pool de envs paralelos (--workers)
  - evolucao com teto curto (--evo-cap 500) + validacao full (--full-steps 1000) no top-3
  - V vem pronto do modelo hibrido (custo zero); M usa --mem-frames reduzidos

Orcamento padrao 100k: mem-frames (5k) + evolucao (92k) + validacao top-3 (3k).
A avaliacao final de 10 episodios (full-steps) e extra, como no v1.

Uso:
  python src/train_worldmodels_cma.py --timesteps 100000 --workers 4 --run-id worldmodels_cma_100k
  python src/train_worldmodels_cma.py --test-run
"""
import argparse
import atexit
import io
import os
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_worldmodels_ga as wm
from env import MarioNdsEnv

# ---------------------------------------------------------------- V: encoders
HYBRID_PREFIX = "features_extractor.encoder."


def load_encoder(source, device, hybrid_path="models/ppo_hybrid_ae_curl_100k.zip",
                 ae_path="models/autoencoder.pth"):
    """Carrega o ENCODER (so ele) e congela. Retorna (vision, tag)."""
    t0 = time.time()
    vision = wm.VisionEncoder().to(device)
    tag = source
    if source == "hybrid" and os.path.exists(hybrid_path):
        print(f"[V] Extraindo encoder hibrido AE+CURL de {hybrid_path} ...", flush=True)
        with zipfile.ZipFile(hybrid_path) as z:
            buf = io.BytesIO(z.read("policy.pth"))
        sd = torch.load(buf, map_location="cpu")
        enc_sd = {k.replace(HYBRID_PREFIX, "encoder."): v
                  for k, v in sd.items() if k.startswith(HYBRID_PREFIX)}
        assert len(enc_sd) == 8, f"chaves inesperadas: {len(enc_sd)}"
        vision.load_state_dict(enc_sd)
        print("[V] Encoder hibrido (pi/vf/shared -> shared) carregado.", flush=True)
    else:
        if source == "hybrid":
            print(f"[V] {hybrid_path} ausente, caindo p/ autoencoder.", flush=True)
            tag = "autoencoder(fallback)"
        vision = wm.load_vision(device, path=ae_path)
        return vision, tag
    for p in vision.parameters():
        p.requires_grad = False
    vision.eval()
    print(f"[V] Vision pronta em {time.time()-t0:.1f}s [{tag}] (frozen, latent=512)", flush=True)
    return vision, tag


# ---------------------------------------------------------------- workers paralelos
_G = {}


def _worker_init(wid_counter, wid_lock, rom, state, enc_state, mem_state):
    import time as _time
    import torch as _t
    # ID unico por worker + inicializacao escalonada: DeSmuMEs simultaneos
    # colidem no Windows (mesmo padrao de train_ppo_recurrent.make_env)
    with wid_lock:
        wid = int(wid_counter.value)
        wid_counter.value = wid + 1
    _t.set_num_threads(1)
    _time.sleep(wid * 4.0)
    tmp = tempfile.gettempdir()
    pid = os.getpid()
    rom_ext = os.path.splitext(rom)[1]
    st_ext = os.path.splitext(state)[1]
    t_rom = os.path.join(tmp, f"mario_rom_cma_{wid}_{pid}{rom_ext}")
    t_st = os.path.join(tmp, f"mario_state_cma_{wid}_{pid}{st_ext}")
    shutil.copy2(rom, t_rom)
    shutil.copy2(state, t_st)
    last_err = None
    for attempt in range(2):
        try:
            _G["env"] = MarioNdsEnv(rom_path=t_rom, state_path=t_st)
            _G["env"].reset()
            last_err = None
            break
        except Exception as e:  # noqa: BLE001 - DeSmuME nativo pode falhar sob contencao
            last_err = e
            _time.sleep(5.0)
    if last_err is not None:
        raise last_err
    _G["vision"] = wm.VisionEncoder()
    _G["vision"].encoder.load_state_dict(enc_state)
    _G["vision"].eval()
    for p in _G["vision"].parameters():
        p.requires_grad = False
    _G["memory"] = wm.MemoryRNN()
    _G["memory"].load_state_dict(mem_state)
    _G["memory"].eval()
    _G["device"] = _t.device("cpu")

    def _cleanup():
        try:
            _G["env"].close()
        except Exception:
            pass
        for f in (t_rom, t_st):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

    atexit.register(_cleanup)


def _eval_vec(vec, max_steps):
    """1 episodio com o controlador linear. Retorna (fitness, steps)."""
    W, b = wm.vec_to_params(np.asarray(vec, dtype=np.float64))
    env, vision, memory, device = _G["env"], _G["vision"], _G["memory"], _G["device"]
    obs, _ = env.reset()
    hx = torch.zeros(1, 1, wm.HIDDEN_DIM)
    cx = torch.zeros(1, 1, wm.HIDDEN_DIM)
    total, steps, done = 0.0, 0, False
    with torch.no_grad():
        while not done and steps < max_steps:
            z = wm.encode_obs(vision, obs, device)
            h = hx.squeeze(0).squeeze(0).numpy()
            action = int(np.argmax(W @ np.concatenate([z, h]) + b))
            obs, reward, done, trunc, _ = env.step(action)
            done = bool(done or trunc)
            zt = torch.from_numpy(z).unsqueeze(0).unsqueeze(0)
            at = torch.zeros(1, 1, wm.N_ACTIONS)
            at[0, 0, action] = 1.0
            _, (hx, cx) = memory.lstm(torch.cat([zt, at], dim=-1), (hx, cx))
            total += float(reward)
            steps += 1
    return float(total), int(steps)


# ---------------------------------------------------------------- sep-CMA-ES
def sep_cma_optimize(pool, dim, budget_steps, evo_cap, pop_size=24, sigma0=0.08,
                     seed=0, mlflow=None):
    """sep-CMA-ES (diagonal). Avaliacoes via pool. Retorna (archive, used, gens, time)."""
    rng = np.random.default_rng(seed)
    lam, mu = pop_size, pop_size // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    n = dim
    cs = (mueff + 2) / (n + mueff + 5)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    m = rng.normal(0, 0.05, size=n)
    sigma, C = sigma0, np.ones(n)
    ps, pc = np.zeros(n), np.zeros(n)
    archive, used, gen = [], 0, 0
    best_fit = -1e18
    t0 = time.time()
    while used < budget_steps:
        gen += 1
        Z = rng.normal(0, 1, size=(lam, n))
        X = m + sigma * Z * np.sqrt(C)
        fits_steps = list(pool.map(_eval_vec, [x for x in X],
                                   [evo_cap] * lam))
        fits = np.array([f for f, _ in fits_steps])
        costs = np.array([c for _, c in fits_steps])
        used += int(costs.sum())
        for x, f in zip(X, fits):
            archive.append((float(f), x.copy()))
            if f > best_fit:
                best_fit = float(f)
        order = np.argsort(-fits)
        m_old = m.copy()
        m = (w[:, None] * X[order[:mu]]).sum(axis=0)
        Y = (X[order[:mu]] - m_old) / sigma
        y_w = (m - m_old) / sigma
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_w / np.sqrt(C)
        norm_ps = float(np.linalg.norm(ps))
        hsig = 1.0 if norm_ps ** 2 / (1 - (1 - cs) ** (2 * gen)) / n < 2 + 4 / (n + 1) else 0.0
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
        C = ((1 - c1 - cmu) * C + c1 * (pc * pc + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (w[:, None] * Y * Y).sum(axis=0))
        C = np.maximum(C, 1e-12)
        sigma = float(np.clip(sigma * np.exp((cs / damps) * (norm_ps / chiN - 1)),
                              1e-4, 1.0))
        print(f"[CMA] gen {gen}: max={fits[order[0]]:.2f} mean={fits.mean():.2f} "
              f"best={best_fit:.2f} sigma={sigma:.4f} steps={used}/{budget_steps} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        if mlflow is not None:
            try:
                mlflow.log_metric("cma_gen_best", float(fits[order[0]]), step=gen)
                mlflow.log_metric("cma_gen_mean", float(fits.mean()), step=gen)
                mlflow.log_metric("cma_sigma", sigma, step=gen)
                mlflow.log_metric("cma_env_steps", int(used), step=gen)
            except Exception:
                pass
    return archive, int(used), gen, time.time() - t0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="World Models v2: encoder hibrido + LSTM + sep-CMA-ES")
    ap.add_argument("--rom", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
    ap.add_argument("--state", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--mem-frames", type=int, default=5000)
    ap.add_argument("--val-reserve", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pop-size", type=int, default=24)
    ap.add_argument("--evo-cap", type=int, default=500)
    ap.add_argument("--full-steps", type=int, default=1000)
    ap.add_argument("--encoder", type=str, default="hybrid", choices=["hybrid", "autoencoder"])
    ap.add_argument("--sigma0", type=float, default=0.08)
    ap.add_argument("--run-id", type=str, default="worldmodels_cma_100k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-run", action="store_true")
    args = ap.parse_args()

    if args.test_run:
        args.timesteps, args.mem_frames, args.val_reserve = 3000, 800, 600
        args.workers, args.pop_size, args.evo_cap, args.full_steps = 2, 6, 200, 300

    assert args.mem_frames + args.val_reserve < args.timesteps
    evo_budget = args.timesteps - args.mem_frames - args.val_reserve
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Orcamento: total={args.timesteps} "
          f"(mem={args.mem_frames} + cma={evo_budget} + valid={args.val_reserve}) "
          f"| workers={args.workers}", flush=True)
    total_t0 = time.time()

    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_NDS_RL")
    with mlflow.start_run(run_name=args.run_id):
        for k in ["model_type", "total_timesteps", "mem_frames", "evo_budget",
                  "workers", "pop_size", "evo_cap", "full_steps", "encoder_src"]:
            mlflow.log_param(k, {"model_type": "WorldModels-sepCMA-ES (hybridEnc+LSTM+linear)",
                                 "total_timesteps": args.timesteps,
                                 "mem_frames": args.mem_frames, "evo_budget": evo_budget,
                                 "workers": args.workers, "pop_size": args.pop_size,
                                 "evo_cap": args.evo_cap, "full_steps": args.full_steps,
                                 "encoder_src": args.encoder}[k])

        # V (custo zero: pesos ja treinados)
        t_v = time.time()
        vision, enc_tag = load_encoder(args.encoder, device)
        v_time = time.time() - t_v

        # M: coleta sequencial c/ 1 env local + treino LSTM na GPU
        ga_env = MarioNdsEnv(rom_path=args.rom, state_path=args.state)
        frames, actions, dones, collect_time = wm.collect_sequential_data(
            ga_env, args.mem_frames, seed=args.seed)
        memory = wm.MemoryRNN()
        mem_loss, mem_train_time = wm.train_memory(
            memory, frames, actions, dones, vision, device, epochs=8)
        mlflow.log_metric("mem_loss_final", mem_loss)
        os.makedirs("models", exist_ok=True)
        torch.save(memory.state_dict(), f"models/{args.run_id}_memory.pth")
        ga_env.close()
        del ga_env, frames

        # C: pool paralelo de envs persistentes (init escalonado anti-colisao)
        from multiprocessing import Lock, Value
        enc_state = {k: v.cpu() for k, v in vision.encoder.state_dict().items()}
        mem_state = {k: v.cpu() for k, v in memory.state_dict().items()}
        dim = (wm.LATENT_DIM + wm.HIDDEN_DIM) * wm.N_ACTIONS + wm.N_ACTIONS
        wid_counter, wid_lock = Value("i", 0), Lock()
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init,
                                 initargs=(wid_counter, wid_lock, args.rom, args.state,
                                           enc_state, mem_state)) as pool:
            archive, evo_steps, gens, cma_time = sep_cma_optimize(
                pool, dim, evo_budget, args.evo_cap, pop_size=args.pop_size,
                sigma0=args.sigma0, seed=args.seed, mlflow=mlflow)

            # Validacao full do top-3 (cap curto pode embaralhar o ranking)
            archive.sort(key=lambda t: -t[0])
            cands = [v for _, v in archive[:3]]
            val = list(pool.map(_eval_vec, cands, [args.full_steps] * len(cands)))
            val_steps = sum(c for _, c in val)
            bi = int(np.argmax([f for f, _ in val]))
            best_vec, best_val = cands[bi], float(val[bi][0])
            print(f"[VAL] top-3@{args.full_steps}: {[f'{f:.2f}' for f, _ in val]} "
                  f"-> campeao={best_val:.2f}", flush=True)
            mlflow.log_metric("val_best_full", best_val)

            # Avaliacao final 10 episodios (protocolo do README), via pool
            print("Avaliacao final (10 episodios)...", flush=True)
            ev = list(pool.map(_eval_vec, [best_vec] * 10,
                               [args.full_steps] * 10))
            eval_steps = sum(c for _, c in ev)
            for i, (f, s) in enumerate(ev):
                print(f"  ep {i+1}: reward={f:.2f} steps={s}", flush=True)

        scores = np.array([f for f, _ in ev])
        print(f"RESULTADO 10eps: media={scores.mean():.2f} std={scores.std():.2f} "
              f"max={scores.max():.2f}", flush=True)

        np.savez(f"models/{args.run_id}.npz", W=wm.vec_to_params(best_vec)[0],
                 b=wm.vec_to_params(best_vec)[1])
        mlflow.log_artifact(f"models/{args.run_id}.npz")
        mlflow.log_artifact(f"models/{args.run_id}_memory.pth")

        total_time = time.time() - total_t0
        train_steps = args.mem_frames + evo_steps + val_steps
        print("=" * 60, flush=True)
        print("TEMPO DE TREINAMENTO (wall-clock):", flush=True)
        print(f"  Encoder (pronto) : {v_time:.1f}s", flush=True)
        print(f"  Coleta memoria   : {collect_time:.1f}s", flush=True)
        print(f"  Treino memoria   : {mem_train_time:.1f}s (GPU)", flush=True)
        print(f"  CMA-ES ({gens} gens, {args.workers} workers): {cma_time:.1f}s", flush=True)
        print(f"  TOTAL            : {total_time:.1f}s = {total_time/60:.1f} min", flush=True)
        print(f"  Env steps treino : {train_steps} (mem={args.mem_frames} + "
              f"cma={evo_steps} + valid={val_steps})", flush=True)
        print(f"  Campeao validado : {best_val:.2f}", flush=True)
        print("=" * 60, flush=True)
        for k, v in [("wall_time_total_s", total_time), ("wall_time_cma_s", cma_time),
                     ("env_steps_train", train_steps),
                     ("eval_mean_10ep", float(scores.mean())),
                     ("eval_std_10ep", float(scores.std())),
                     ("eval_max_10ep", float(scores.max()))]:
            mlflow.log_metric(k, v)


if __name__ == "__main__":
    main()
