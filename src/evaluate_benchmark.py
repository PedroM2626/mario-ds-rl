"""
Benchmark unificado (protocolo novo):
  10 episodios DETERMINISTICOS + 10 ESTOCASTICOS por modelo,
  episodios completos (sem render), UM unico env reutilizado.
  Auto-detecta RecurrentPPO (single frame + LSTM) vs PPO (VecFrameStack).

Uso:
  python src/evaluate_benchmark.py [--models ae_frozen,hybrid,recurrent,impala,pure,spr,curl,drq,icm,icm_ae]
  python src/evaluate_benchmark.py --models icm,icm_ae   # apos treinar o item B
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
MAX_STEPS = 5000  # teto de seguranca; episodio natural termina em morte/fim/timeout

MODELS = {
    "pure":      "models/ppo_pure_100k.zip",
    "ae_frozen": "models/ppo_autoencoder_frozen_100k.zip",
    "curl":      "models/ppo_curl_online_100k.zip",
    "hybrid":    "models/ppo_hybrid_ae_curl_100k.zip",
    "spr":       "models/ppo_spr_100k.zip",
    "drq":       "models/ppo_drq_100k.zip",
    "recurrent": "models/recurrent_ppo_mario.zip",
    "impala":    "models/mario_impala.zip",
    "icm":       "models/ppo_icm_100k.zip",
    "icm_ae":    "models/ppo_icm_ae_100k.zip",
}


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor as _BFE


class _DrQPassthrough(_BFE):
    """Replica a estrutura do DrQFeaturesExtractorWrapper (train_ppo_recurrent,
    function-local e por isso nao reconstruivel no load). Em eval (training=False)
    o original e passthrough puro, logo a inferencia e identica."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=512)
        from stable_baselines3.common.torch_layers import NatureCNN
        self.base_extractor = NatureCNN(observation_space)

    def forward(self, observations):
        return self.base_extractor(observations)


def _load_drq_manual(path):
    """Fallback p/ modelos DrQ: constroi RecurrentPPO com kwargs conhecidos
    (policy_kwargs salvos nao incluem o wrapper, trocado pos-construcao no treino)
    e carrega policy.pth direto na policy."""
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
                       # sem isto o SB3 resolve net_arch=None p/ [64,64] quando
                       # ha extrator custom; o treino DrQ usava [] (default sem
                       # extrator custom, que era trocado pos-construcao)
                       "net_arch": [],
                       "lstm_hidden_size": 256,
                       "enable_critic_lstm": True})
    # No treino, SÓ o extrator compartilhado foi trocado pelo wrapper DrQ;
    # pi_/vf_ seguiram NatureCNN puros (não usados no forward com share=True,
    # mas presentes no state_dict). Replica a estrutura p/ load estrito.
    from stable_baselines3.common.torch_layers import NatureCNN
    model.policy.pi_features_extractor = NatureCNN(model.observation_space)
    model.policy.vf_features_extractor = NatureCNN(model.observation_space)
    model.policy.to(model.device)
    sd = torch.load(io.BytesIO(raw), map_location=model.device)
    model.policy.load_state_dict(sd, strict=True)
    return model


def load_model(path):
    """Retorna (model, kind). kind = 'recurrent' | 'ppo'."""
    err_rec = err_ppo = None
    try:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path), "recurrent"
    except Exception as e:
        err_rec = e
    if "base_extractor" in str(err_rec):
        # modelo treinado com DrQ wrapper (trocado pos-construcao no treino,
        # logo ausente nos policy_kwargs salvos): construcao manual + policy.pth
        try:
            from sb3_contrib import RecurrentPPO  # noqa: F401 (garante import)
            return _load_drq_manual(path), "recurrent"
        except Exception as e:
            err_rec = e
    try:
        from stable_baselines3 import PPO
        return PPO.load(path), "ppo"
    except Exception as e:
        err_ppo = e
    raise RuntimeError(f"Falha ao carregar {path} como RecurrentPPO ({err_rec}) "
                       f"e como PPO ({err_ppo})")


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
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=",".join(MODELS),
                    help="subset separado por virgula")
    ap.add_argument("--eps", type=int, default=10)
    args = ap.parse_args()
    names = [m.strip() for m in args.models.split(",") if m.strip()]

    results = {}
    # NUNCA dois DeSmuMEs vivos no mesmo processo (colisao nativa):
    # fase recurrent com env cru, depois fecha e abre a fase framestack.
    rec_names = [n for n in names if MODELS[n] != MODELS.get("impala")]
    stk_names = [n for n in names if MODELS[n] == MODELS.get("impala")]
    t0 = time.time()
    env = None
    if rec_names:
        env = MarioNdsEnv(rom_path=ROM, state_path=STATE)  # unico env p/ recurrent
        assert env.has_emulator, "emulador falhou no env principal!"
    try:
        for name in rec_names:
            path = MODELS[name]
            if not os.path.exists(path):
                print(f"[{name}] SKIP (sem arquivo: {path})", flush=True)
                continue
            model, kind = load_model(path)
            assert kind == "recurrent", f"{name} nao e recurrent!"
            print(f"[{name}] {kind} policy={type(model.policy).__name__} "
                  f"obs={model.observation_space.shape}", flush=True)
            out = {"kind": kind, "path": path}
            for mode, det in (("det", True), ("stoch", False)):
                r, s = eval_recurrent(model, env, args.eps, det)
                out[mode] = {"rewards": r, "steps": s,
                             "mean": float(np.mean(r)), "std": float(np.std(r)),
                             "max": float(np.max(r)),
                             "mean_steps": float(np.mean(s))}
                print(f"[{name}/{mode}] media={np.mean(r):.2f} std={np.std(r):.2f} "
                      f"max={np.max(r):.2f} steps_medios={np.mean(s):.0f}", flush=True)
            results[name] = out
            del model
    finally:
        if env is not None:
            env.close()

    for name in stk_names:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
        path = MODELS[name]
        if not os.path.exists(path):
            print(f"[{name}] SKIP (sem arquivo: {path})", flush=True)
            continue
        model, kind = load_model(path)
        assert kind == "ppo", f"{name} nao e ppo/framestack!"
        venv = MarioNdsEnv(rom_path=ROM, state_path=STATE)
        assert venv.has_emulator, "emulador falhou no env framestack!"
        vec_env = VecFrameStack(DummyVecEnv([lambda: venv]), n_stack=4)
        try:
            print(f"[{name}] {kind} policy={type(model.policy).__name__} "
                  f"obs={model.observation_space.shape}", flush=True)
            out = {"kind": kind, "path": path}
            for mode, det in (("det", True), ("stoch", False)):
                r, s = eval_ppo_framestack(model, vec_env, args.eps, det)
                out[mode] = {"rewards": r, "steps": s,
                             "mean": float(np.mean(r)), "std": float(np.std(r)),
                             "max": float(np.max(r)),
                             "mean_steps": float(np.mean(s))}
                print(f"[{name}/{mode}] media={np.mean(r):.2f} std={np.std(r):.2f} "
                      f"max={np.max(r):.2f} steps_medios={np.mean(s):.0f}", flush=True)
            results[name] = out
        finally:
            vec_env.close()

    os.makedirs("evals", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fp = f"evals/benchmark_20ep_{ts}.json"
    with open(fp, "w") as f:
        json.dump({"elapsed_s": time.time() - t0, "results": results}, f, indent=1)
    print(f"RESULTADOS salvos em {fp} ({time.time()-t0:.0f}s)", flush=True)
    print("TABELA det | stoch (media +- std, max):", flush=True)
    for name, o in results.items():
        print(f"  {name}: det={o['det']['mean']:.2f}+-{o['det']['std']:.2f}"
              f" (max {o['det']['max']:.2f}) | stoch={o['stoch']['mean']:.2f}+-"
              f"{o['stoch']['std']:.2f} (max {o['stoch']['max']:.2f})", flush=True)


if __name__ == "__main__":
    main()
