"""
World Models (Ha & Schmidhuber, 2018) simplificado para Mario DS — 3 camadas:
  V (Vision/Encoder): Encoder CNN congelado pre-treinado VIA Autoencoder
      (models/autoencoder.pth, latente 512). O Autoencoder (encoder+decoder)
      foi so o metodo de pre-treino por reconstrucao; o decoder e descartado
      e so o ENCODER vira extrator de features visuais para o RL.
  M (Memory):  LSTM que prediz o proximo latente z_{t+1} dado (z_t, a_t)
  C (Controller): controlador linear minisculo evoluido com Algoritmo Genetico

Orcamento total padrao: 100k env steps (comparavel aos baselines PPO_*_100k),
divididos em:
  --mem-frames : steps de coleta aleatoria p/ treinar a memoria (default 10k)
  restante     : avaliacoes do GA (cada episodio conta no budget)

Uso:
  python src/train_worldmodels_ga.py --timesteps 100000 --mem-frames 10000 --pop-size 24 --run-id worldmodels_ga_100k
  python src/train_worldmodels_ga.py --test-run   # sanity check rapido (~3k steps)
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import MarioNdsEnv

LATENT_DIM = 512
HIDDEN_DIM = 256
N_ACTIONS = 6


# ---------------------------------------------------------------- V: Vision
class VisionEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


def load_vision(device, path="models/autoencoder.pth"):
    t0 = time.time()
    model = VisionEncoder().to(device)
    if os.path.exists(path):
        print(f"[V] Carregando pesos do Autoencoder: {path}")
        sd = torch.load(path, map_location=device)
        enc_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_sd)
    else:
        print(f"[V] AVISO: {path} nao encontrado. Usando encoder aleatorio.")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print(f"[V] Vision pronta em {time.time()-t0:.1f}s (frozen, latent={LATENT_DIM})")
    return model


@torch.no_grad()
def encode_obs(vision, obs, device):
    """obs: (84,84,1) uint8 -> z: (512,) float32 numpy"""
    x = torch.from_numpy(obs.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    z = vision(x).squeeze(0).float().cpu().numpy()
    return z


# ---------------------------------------------------------------- M: Memory
class MemoryRNN(nn.Module):
    """LSTM que modela p(z_{t+1} | z_t, a_t). Camada de memoria do World Models."""

    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(latent_dim + n_actions, hidden_dim, batch_first=True)
        self.pred_z = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_seq, a_seq, hidden=None):
        # z_seq: (B,T,512), a_seq: (B,T) int64
        a_onehot = torch.nn.functional.one_hot(a_seq, num_classes=N_ACTIONS).float()
        x = torch.cat([z_seq, a_onehot], dim=-1)
        out, hidden = self.lstm(x, hidden)
        return self.pred_z(out), hidden


def collect_sequential_data(env, num_frames, seed=0):
    """Coleta frames SEQUENCIAIS com politica aleatoria (necessario p/ treinar LSTM).
    Recebe um env JA CRIADO e o reutiliza (criar 2+ DeSmuMEs no mesmo processo
    causa access violation). Nao fecha o env."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    frames, actions, dones = [], [], []
    obs, _ = env.reset()
    for _ in range(num_frames):
        r = rng.random()
        if r < 0.6:
            a = int(rng.choice([1, 2, 3]))
        else:
            a = int(rng.integers(0, N_ACTIONS))
        frames.append(obs.copy())
        actions.append(a)
        obs, _, done, trunc, _ = env.step(a)
        d = bool(done or trunc)
        dones.append(d)
        if d:
            obs, _ = env.reset()
    elapsed = time.time() - t0
    print(f"[M] Coleta sequencial: {num_frames} frames em {elapsed:.1f}s "
          f"({num_frames/max(elapsed,1e-6):.1f} fps)")
    return np.array(frames), np.array(actions, dtype=np.int64), np.array(dones, dtype=bool), elapsed


def train_memory(memory, frames, actions, dones, vision, device, epochs=8, seq_len=32, batch_size=32, lr=1e-3):
    """Treina LSTM a prever z_{t+1}. Retorna (loss_final, tempo_s)."""
    t0 = time.time()
    print("[M] Codificando frames para latentes (GPU, em batch)...")
    vision.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(frames), 512):
            b = frames[i:i + 512].astype(np.float32) / 255.0
            b = np.transpose(b, (0, 3, 1, 2))
            t = torch.from_numpy(b).to(device)
            zs.append(vision(t).float().cpu())
    z_all = torch.cat(zs, dim=0)  # (N,512)
    a_all = torch.from_numpy(actions)

    # Monta sequencias que nao cruzam episodio (done)
    seqs_z, seqs_a = [], []
    i = 0
    N = len(frames)
    while i + seq_len + 1 < N:
        if dones[i:i + seq_len + 1].any():
            i += 1
            continue
        seqs_z.append(z_all[i:i + seq_len + 1])
        seqs_a.append(a_all[i:i + seq_len + 1])
        i += seq_len
    if not seqs_z:
        print("[M] AVISO: poucas sequencias limpas, usando janelas com corte.")
        for i in range(0, N - seq_len - 1, seq_len):
            seqs_z.append(z_all[i:i + seq_len + 1])
            seqs_a.append(a_all[i:i + seq_len + 1])
    seqs_z = torch.stack(seqs_z)  # (S,T+1,512)
    seqs_a = torch.stack(seqs_a)  # (S,T+1)
    print(f"[M] {len(seqs_z)} sequencias de len={seq_len+1} para treino do LSTM.")

    memory.train().to(device)
    opt = torch.optim.Adam(memory.parameters(), lr=lr)
    crit = nn.MSELoss()
    n = len(seqs_z)
    final_loss = float("nan")
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for j in range(0, n, batch_size):
            idx = perm[j:j + batch_size]
            zb = seqs_z[idx].to(device)          # (B,T+1,512)
            ab = seqs_a[idx].to(device)          # (B,T+1)
            pred, _ = memory(zb[:, :-1], ab[:, :-1])
            loss = crit(pred, zb[:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        final_loss = tot / max(nb, 1)
        print(f"[M] epoch {ep+1}/{epochs} loss={final_loss:.6f}")
    memory.eval()
    elapsed = time.time() - t0
    print(f"[M] Memoria treinada em {elapsed:.1f}s. loss_final={final_loss:.6f}")
    return final_loss, elapsed


# ---------------------------------------------------------------- C: Controller (GA)
def vec_to_params(vec, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, n_actions=N_ACTIONS):
    d = latent_dim + hidden_dim
    W = vec[: d * n_actions].reshape(n_actions, d)
    b = vec[d * n_actions: d * n_actions + n_actions]
    return W, b


def log(msg):
    print(msg, flush=True)


def evaluate_individual(vec, env, vision, memory, device, max_steps=1000):
    """Roda 1 episodio com o controlador linear num env REUTILIZADO.
    Retorna (fitness, steps_usados). Reutilizar o mesmo env evita
    recriar o DeSmuME (que causa access violation em criacao/destruicao repetida)."""
    W, b = vec_to_params(vec)
    obs, _ = env.reset()
    hx = torch.zeros(1, 1, HIDDEN_DIM, device=device)
    cx = torch.zeros(1, 1, HIDDEN_DIM, device=device)
    total, steps = 0.0, 0
    done = False
    with torch.no_grad():
        while not done and steps < max_steps:
            z = encode_obs(vision, obs, device)              # (512,)
            h = hx.squeeze(0).squeeze(0).cpu().numpy()       # (256,)
            x = np.concatenate([z, h])
            logits = W @ x + b
            action = int(np.argmax(logits))
            obs, reward, done, trunc, _ = env.step(action)
            done = bool(done or trunc)
            # atualiza memoria: LSTM com seq len 1
            zt = torch.from_numpy(z).unsqueeze(0).unsqueeze(0).to(device)
            at = torch.zeros(1, 1, N_ACTIONS, device=device)
            at[0, 0, action] = 1.0
            _, (hx, cx) = memory.lstm(torch.cat([zt, at], dim=-1), (hx, cx))
            total += float(reward)
            steps += 1
    return total, steps


def evolve(env, vision, memory, device, budget_steps, pop_size=24, max_steps=1000,
           sigma=0.05, elite_frac=0.25, seed=0, mlflow=None):
    rng = np.random.default_rng(seed)
    d = (LATENT_DIM + HIDDEN_DIM) * N_ACTIONS + N_ACTIONS
    pop = rng.normal(0, 0.1, size=(pop_size, d)).astype(np.float64)
    n_elite = max(1, int(pop_size * elite_frac))
    best_vec, best_fit = pop[0].copy(), -1e18
    used, gen = 0, 0
    history = []
    t0 = time.time()
    while used < budget_steps:
        gen += 1
        fits, costs = [], []
        for i in range(pop_size):
            if used >= budget_steps:
                fits.append(-1e18)
                costs.append(0)
                continue
            f, c = evaluate_individual(pop[i], env, vision, memory, device, max_steps)
            fits.append(f)
            costs.append(c)
            used += c
            if f > best_fit:
                best_vec, best_fit = pop[i].copy(), f
        fits = np.array(fits)
        valid = fits > -1e17
        mean_f = fits[valid].mean() if valid.any() else float("nan")
        max_f = fits[valid].max() if valid.any() else float("nan")
        history.append((gen, float(max_f), float(mean_f), int(used)))
        print(f"[C] gen {gen}: max={max_f:.2f} mean={mean_f:.2f} "
              f"best_global={best_fit:.2f} steps={used}/{budget_steps} "
              f"({(time.time()-t0)/60:.1f} min)")
        if mlflow is not None:
            try:
                mlflow.log_metric("ga_gen_best", float(max_f), step=gen)
                mlflow.log_metric("ga_gen_mean", float(mean_f), step=gen)
                mlflow.log_metric("ga_best_global", float(best_fit), step=gen)
                mlflow.log_metric("ga_env_steps", int(used), step=gen)
            except Exception:
                pass
        if used >= budget_steps:
            break
        # selecao + mutacao gaussiana (elitismo)
        order = np.argsort(-np.where(valid, fits, -1e18))
        elites = pop[order[:n_elite]]
        children = [elites[0].copy()]  # clone do campeao
        while len(children) < pop_size:
            parent = elites[rng.integers(0, n_elite)]
            children.append(parent + rng.normal(0, sigma, size=d))
        pop = np.array(children)
    ga_time = time.time() - t0
    return best_vec, float(best_fit), int(used), ga_time, history


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="World Models GA: VAE + LSTM + Algoritmo Genetico")
    ap.add_argument("--rom", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds")
    ap.add_argument("--state", type=str, default="data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1")
    ap.add_argument("--timesteps", type=int, default=100000, help="Orcamento TOTAL de env steps (M coleta + GA)")
    ap.add_argument("--mem-frames", type=int, default=10000, help="Steps p/ treino da memoria")
    ap.add_argument("--mem-epochs", type=int, default=8)
    ap.add_argument("--pop-size", type=int, default=24)
    ap.add_argument("--max-ep-steps", type=int, default=1000)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--run-id", type=str, default="worldmodels_ga_100k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-run", action="store_true")
    args = ap.parse_args()

    if args.test_run:
        args.timesteps = 3000
        args.mem_frames = 1000
        args.mem_epochs = 1
        args.pop_size = 6
        args.max_ep_steps = 300

    assert args.mem_frames < args.timesteps, "--mem-frames deve ser menor que --timesteps"
    ga_budget = args.timesteps - args.mem_frames

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Orcamento total: {args.timesteps} steps "
          f"(memoria={args.mem_frames} + GA={ga_budget})")
    total_t0 = time.time()

    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_NDS_RL")
    with mlflow.start_run(run_name=args.run_id):
        mlflow.log_param("model_type", "WorldModels-GA (V+LSTM+linear)")
        mlflow.log_param("total_timesteps", args.timesteps)
        mlflow.log_param("mem_frames", args.mem_frames)
        mlflow.log_param("ga_budget", ga_budget)
        mlflow.log_param("pop_size", args.pop_size)
        mlflow.log_param("latent_dim", LATENT_DIM)
        mlflow.log_param("hidden_dim", HIDDEN_DIM)

        # UM unico env para o processo inteiro (2+ DeSmuMEs = crash nativo)
        ga_env = MarioNdsEnv(rom_path=args.rom, state_path=args.state)

        # V
        vision = load_vision(device)

        # M
        frames, actions, dones, collect_time = collect_sequential_data(
            ga_env, args.mem_frames, seed=args.seed)
        memory = MemoryRNN()
        mem_loss, mem_train_time = train_memory(
            memory, frames, actions, dones, vision, device, epochs=args.mem_epochs)
        mlflow.log_metric("mem_loss_final", mem_loss)
        os.makedirs("models", exist_ok=True)
        torch.save(memory.state_dict(), f"models/{args.run_id}_memory.pth")

        # C — reusa o mesmo env
        best_vec, best_fit, ga_steps, ga_time, hist = evolve(
            ga_env, vision, memory, device, ga_budget,
            pop_size=args.pop_size, max_steps=args.max_ep_steps,
            sigma=args.sigma, seed=args.seed, mlflow=mlflow)
        mlflow.log_metric("ga_best_fitness_train", best_fit)
        np.savez(f"models/{args.run_id}.npz", W=vec_to_params(best_vec)[0],
                 b=vec_to_params(best_vec)[1], latent_dim=LATENT_DIM,
                 hidden_dim=HIDDEN_DIM)
        mlflow.log_artifact(f"models/{args.run_id}.npz")
        mlflow.log_artifact(f"models/{args.run_id}_memory.pth")

        total_time = time.time() - total_t0
        total_steps = args.mem_frames + ga_steps
        print("\n" + "=" * 60)
        print(f"TEMPO DE TREINAMENTO (wall-clock):")
        print(f"  Coleta memoria : {collect_time:.1f}s")
        print(f"  Treino memoria : {mem_train_time:.1f}s (GPU)")
        print(f"  Evolucao GA    : {ga_time:.1f}s")
        print(f"  TOTAL          : {total_time:.1f}s = {total_time/60:.1f} min")
        print(f"  Env steps usados: {total_steps} (mem={args.mem_frames} + ga={ga_steps})")
        print(f"  Melhor fitness (treino, 1 ep): {best_fit:.2f}")
        print("=" * 60)
        mlflow.log_metric("wall_time_total_s", total_time)
        mlflow.log_metric("wall_time_ga_s", ga_time)
        mlflow.log_metric("wall_time_mem_train_s", mem_train_time)
        mlflow.log_metric("env_steps_total", total_steps)

        # Avaliacao final: 10 episodios (mesmo protocolo do README), reusa ga_env
        print("\nAvaliacao final (10 episodios)...", flush=True)
        scores = []
        for ep in range(10):
            f, s = evaluate_individual(best_vec, ga_env, vision,
                                       memory, device, max_steps=args.max_ep_steps)
            scores.append(f)
            print(f"  ep {ep+1}: reward={f:.2f} steps={s}", flush=True)
        ga_env.close()
        scores = np.array(scores)
        print(f"RESULTADO 10eps: media={scores.mean():.2f} std={scores.std():.2f} max={scores.max():.2f}")
        mlflow.log_metric("eval_mean_10ep", float(scores.mean()))
        mlflow.log_metric("eval_std_10ep", float(scores.std()))
        mlflow.log_metric("eval_max_10ep", float(scores.max()))


if __name__ == "__main__":
    main()
